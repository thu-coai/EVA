# coding=utf-8

"""Finetune EVA"""

import os
import json
import torch
import mpu

import torch.distributed as dist

from torch.utils.data import DataLoader, SequentialSampler
from arguments import get_args
from tokenization_eva import EVATokenizer

from utils import save_checkpoint, load_checkpoint
from utils import print_args, print_rank_0, save_rank_0
from utils import set_random_seed, initialize_distributed, set_deepspeed_activation_checkpointing
from model import EVAModel, EVAConfig, enc_dec_get_params_for_weight_decay_optimization
from samplers import DistributedBatchSampler, RandomSampler

from fp16 import FP16_Module, FP16_Optimizer
from learning_rates import AnnealingLR

import deepspeed
from apex.optimizers import FusedAdam as Adam

from generation_metrics import Metric
from generation_utils import generate_beam, generate_no_beam

from eva_datasets import EVADataset
from tqdm import tqdm

import signal
signal.signal(signal.SIGCHLD, signal.SIG_IGN)

from model import DistributedDataParallel as DDP


def get_model(args, config):
    """Build the model."""

    print_rank_0('building Enc-Dec model ...')

    model = EVAModel(
        config,
        parallel_output=True,
        checkpoint_activations=args.checkpoint_activations,
        checkpoint_num_layers=args.checkpoint_num_layers
    )

    if mpu.get_data_parallel_rank() == 0:
        print(' > number of parameters on model parallel rank {}: {}'.format(
            mpu.get_model_parallel_rank(),
            sum([p.nelement() for p in model.parameters()])), flush=True)

    # To prevent OOM for model sizes that cannot fit in GPU memory in full precision
    if args.deepspeed and args.fp16:
        model.half()

    # GPU allocation.
    model.cuda(torch.cuda.current_device())

    # Fp16 conversion.
    if args.fp16:
        model = FP16_Module(model)

    model = DDP(model)

    return model


def get_optimizer(model, args):
    """Set up the optimizer."""

    # Build parameter groups (weight decay and non-decay).
    while isinstance(model, (DDP, FP16_Module)):
        model = model.module
    param_groups = enc_dec_get_params_for_weight_decay_optimization(model)

    # Add model parallel attribute if it is not set.
    for param_group in param_groups:
        for param in param_group['params']:
            if not hasattr(param, 'model_parallel'):
                param.model_parallel = False

    # Use FusedAdam.
    optimizer = Adam(param_groups,
                        lr=args.lr, weight_decay=args.weight_decay)

    print(f'Optimizer = {optimizer.__class__.__name__}')
    if args.deepspeed:
        # fp16 wrapper is not required for DeepSpeed.
        return optimizer

    # Wrap into fp16 optimizer.
    if args.fp16:
        optimizer = FP16_Optimizer(optimizer,
                                   static_loss_scale=args.loss_scale,
                                   dynamic_loss_scale=args.dynamic_loss_scale,
                                   dynamic_loss_args={
                                       'scale_window': args.loss_scale_window,
                                       'min_scale': args.min_scale,
                                       'delayed_shift': args.hysteresis})

    return optimizer


def get_learning_rate_scheduler(optimizer, args):
    """Build the learning rate scheduler."""

    num_iters = args.train_iters
    num_iters = max(1, num_iters)
    init_step = -1
    warmup_iter = args.warmup * num_iters
    lr_scheduler = AnnealingLR(optimizer,
                               start_lr=args.lr,
                               warmup_iter=warmup_iter,
                               num_iters=num_iters,
                               decay_style=args.lr_decay_style,
                               last_iter=init_step,
                               gradient_accumulation_steps=args.gradient_accumulation_steps)

    return lr_scheduler


def setup_model_and_optimizer(args, model_config, ds_config, do_train=True):
    """Setup model and optimizer."""

    model = get_model(args, model_config)
    optimizer, lr_scheduler = None, None
    if do_train:
        optimizer = get_optimizer(model, args)
        lr_scheduler = get_learning_rate_scheduler(optimizer, args)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            args=args,
            lr_scheduler=lr_scheduler,
            mpu=mpu,
            dist_init_required=False,
            config_params=ds_config
        )

    if args.load is not None:
        args.iteration = load_checkpoint(args, model, optimizer, lr_scheduler)
    else:
        args.iteration = 0

    return model, optimizer, lr_scheduler


def load_data(args, data_type, tokenizer, ratio=1, drop_last=True):
    data_path = os.path.join(args.data_path, data_type + args.data_ext)

    # Data parallel arguments.
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if data_type == "train":
        global_batch_size = args.batch_size * world_size
    else:
        global_batch_size = args.eval_batch_size * world_size

    num_workers = args.num_workers

    dataset = EVADataset(
        args,
        tokenizer,
        data_path,
        data_type,
        ratio=ratio,
        cache_path=args.cache_path)

    if data_type == 'train':
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    batch_sampler = DistributedBatchSampler(sampler=sampler,
                                            batch_size=global_batch_size,
                                            drop_last=drop_last,
                                            rank=rank,
                                            world_size=world_size)

    data_loader = DataLoader(dataset,
                             batch_sampler=batch_sampler,
                             num_workers=num_workers,
                             pin_memory=True,
                             collate_fn=dataset.collate)

    # Torch dataloader.
    return data_loader, dataset, sampler


def forward_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=False):
    if keep_enc_hidden:
        enc_outputs = model(**model_batch, only_encoder=True)
        enc_hidden_states = enc_outputs["encoder_last_hidden_state"]
        output = model(**model_batch, enc_hidden_states=enc_hidden_states)
    else:
        output = model(**model_batch)
    
    logits = output["lm_logits"]
    forw_out = {
        "logits": logits
    }
    if keep_enc_hidden:
        forw_out["enc_hidden_states"] = enc_hidden_states
    
    losses = mpu.vocab_parallel_cross_entropy(logits.contiguous().float(), no_model_batch["labels"])

    loss_mask = no_model_batch["loss_mask"]
    losses = (losses * loss_mask).sum(-1) / loss_mask.sum(-1)
    loss = losses.mean()

    forw_out["loss"] = loss
    forw_out["loss_batch"] = losses
    
    return forw_out


def backward_step(args, loss, model, optimizer):
    # backward
    if args.deepspeed:
        model.backward(loss)
    else:
        optimizer.zero_grad()
        if args.fp16:
            optimizer.backward(loss, update_master_grads=False)
        else:
            loss.backward()

    # Update master gradients.
    if not args.deepspeed:
        if args.fp16:
            optimizer.update_master_grads()

        # Clipping gradients helps prevent the exploding gradient.
        if args.clip_grad > 0:
            if not args.fp16:
                mpu.clip_grad_norm(model.parameters(), args.clip_grad)
            else:
                optimizer.clip_master_grads(args.clip_grad)


def train(args, tokenizer, model, optimizer, lr_scheduler, train_dataset, train_dataloader, dev_dataset, dev_dataloader, device):
    """Train the model."""

    # Turn on training mode which enables dropout.
    model.train()

    # Tracking loss.
    total_loss = 0.0

    step, global_step = 1, 1

    for e in range(args.epochs):
        model.train()
        for model_batch, no_model_batch in train_dataloader:
            for k in model_batch:
                model_batch[k] = model_batch[k].to(device)
            for k in no_model_batch:
                no_model_batch[k] = no_model_batch[k].to(device)
            forw_out = forward_step(args, model_batch, no_model_batch, model, device)
            loss = forw_out["loss"]
            
            if torch.distributed.get_rank() == 0:
                print(loss)

            backward_step(args, loss, model, optimizer)

            # Update losses.
            total_loss += loss.item()

            if args.deepspeed:
                model.step()
            else:
                optimizer.step()
                if not (args.fp16 and optimizer.overflow):
                    lr_scheduler.step()

            # Logging.
            if global_step % args.log_interval == 0 and step % args.gradient_accumulation_steps == 0:
                learning_rate = optimizer.param_groups[0]['lr']
                avg_lm_loss = total_loss / (args.log_interval * args.gradient_accumulation_steps)
                log_string = 'epoch {:3d}/{:3d} |'.format(e, args.epochs)
                log_string += ' global iteration {:8d}/{:8d} |'.format(global_step, args.train_iters)
                log_string += ' learning rate {:.3} |'.format(learning_rate)
                log_string += ' lm loss {:.6} |'.format(avg_lm_loss)
                if args.fp16:
                    log_string += ' loss scale {:.1f} |'.format(optimizer.cur_scale if args.deepspeed else optimizer.loss_scale)
                print_rank_0(log_string)
                save_rank_0(args, log_string)
                total_loss = 0.0

            # Checkpointing
            if args.save and args.save_interval and global_step % args.save_interval == 0 and step % args.gradient_accumulation_steps == 0:
                save_checkpoint(global_step, model, optimizer, lr_scheduler, args)

            # Evaluation
            if args.eval_interval and global_step % args.eval_interval == 0 and step % args.gradient_accumulation_steps == 0 and args.do_valid:
                prefix = 'iteration {} | '.format(global_step)
                eval_loss, metric_res, _ = evaluate(args, tokenizer, dev_dataset, dev_dataloader, model, device, mode="dev")
                model.train()
                if len(metric_res) > 1:
                    log_string = prefix
                    for key, value in metric_res.items():
                        log_string += " {}: {:.5} | ".format(key, value)
                else:
                    log_string = prefix + " eval_loss: " + str(eval_loss)
                print_rank_0(log_string)
                save_rank_0(args, log_string)

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1

    # end train
    if args.save:
        save_checkpoint(global_step, model, optimizer, lr_scheduler, args)

    return global_step


def gen_metric(args, tokenizer: EVATokenizer, all_preds, all_labels):
    print("Doing gen metric")
    metric = Metric(tokenizer)
    for l, p in zip(all_labels, all_preds):
        l = list(tokenizer.decode(l[1:-1]))
        p = list(tokenizer.decode(p[1:-1]))
        metric.forword([list(map(str, l))], list(map(str, p)))
    
    metric_res, *_ = metric.close()

    with open(os.path.join(args.save, "{}.txt".format(metric_res["rouge-l"])), "w") as f:
        for p, l in zip(all_preds, all_labels):
            f.write(str(p) + "\t\t" + str(l) + "\n")
            f.write(tokenizer.decode(p) + "\t\t" + tokenizer.decode(l) + "\n\n")

    return metric_res


def evaluate(args, tokenizer, eval_dataset, eval_data_loader, model, device, mode="dev"):
    """Evaluation."""

    model.eval()

    metric = Metric(tokenizer)

    loss_res = None
    generation_res = None
    metric_res = {}
    
    step = 0

    with torch.no_grad():
        loss_res = 0.0
        for model_batch, no_model_batch in eval_data_loader:
            for k in model_batch:
                model_batch[k] = model_batch[k].to(device)
            for k in no_model_batch:
                no_model_batch[k] = no_model_batch[k].to(device)
            forw_out = forward_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=True)
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            loss_res += loss
            step += 1

        loss_res /= step

        if args.eval_generation:
            generation_res = []
            for e, (model_batch, no_model_batch) in enumerate(tqdm(eval_data_loader, desc="Evaluating")):
                for k in model_batch:
                    model_batch[k] = model_batch[k].to(device)
                for k in no_model_batch:
                    no_model_batch[k] = no_model_batch[k].to(device)
                if args.num_beams == 1:
                    generation_str_list, generation_id_list = generate_no_beam(model_batch, model_batch["enc_input_ids"], model, tokenizer, args, device)
                else:
                    generation_str_list, generation_id_list = generate_beam(model_batch, model_batch["enc_input_ids"], model, tokenizer, args, device)

                output_ids = [x + [tokenizer.sep_id] + (args.max_generation_length - len(x)) * [tokenizer.pad_id] for x in generation_id_list]
                output_ids = torch.tensor(output_ids).to(device)

                tmp_labels = [torch.zeros_like(no_model_batch["labels"]).to(device) for _ in range(mpu.get_data_parallel_world_size())]
                torch.distributed.all_gather(tmp_labels, no_model_batch["labels"].data, group=mpu.get_data_parallel_group())

                tmp_output_ids = [torch.zeros_like(output_ids).to(device) for _ in range(mpu.get_data_parallel_world_size())]
                torch.distributed.all_gather(tmp_output_ids, output_ids.data, group=mpu.get_data_parallel_group())
                
                tmp_contexts = [torch.zeros_like(model_batch["enc_input_ids"]).to(device) for _ in range(mpu.get_data_parallel_world_size())]
                torch.distributed.all_gather(tmp_contexts, model_batch["enc_input_ids"].data, group=mpu.get_data_parallel_group())
                
                context_token_ids = sum([e.cpu().tolist() for e in tmp_contexts], [])
                context_token_ids = [e[:e.index(tokenizer.pad_id)] if tokenizer.pad_id in e else e for e in context_token_ids]
        
                label_token_ids = sum([e.cpu().tolist() for e in tmp_labels], [])
                label_token_ids = [e[:e.index(tokenizer.sep_id)] if tokenizer.sep_id in e else e for e in label_token_ids]

                generation_token_ids = sum([e.cpu().tolist() for e in tmp_output_ids], [])
                generation_token_ids = [e[:e.index(tokenizer.sep_id)] if tokenizer.sep_id in e else e for e in generation_token_ids]
                for lab, gen in zip(label_token_ids, generation_token_ids):
                    #metric.forword([list(map(str, lab))], list(map(str, gen)))
                    metric.forstr([tokenizer.decode(lab)], tokenizer.decode(gen))
                
                for ctx, lab, gen in zip(context_token_ids, label_token_ids, generation_token_ids):
                    generation_res.append({
                        'context': tokenizer.decode(ctx),
                        'response': tokenizer.decode(lab),
                        'generation': tokenizer.decode(gen),
                    })
                    if e == 0:
                        print_rank_0(f'****** context: {tokenizer.decode(ctx)}\n'
                                    f'****** response: {tokenizer.decode(lab)}\n'
                                    f'****** generation: {tokenizer.decode(gen)}\n')

            metric_res, *_ = metric.close()

        metric_res["loss"] = loss_res

    return loss_res, metric_res, generation_res


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()

    os.makedirs(args.save, exist_ok=True)
    config = EVAConfig.from_json_file(args.model_config)

    # Pytorch distributed.
    initialize_distributed(args)

    # Optional DeepSpeed Activation Checkpointing Features
    num_checkpoints = config.num_layers // args.checkpoint_num_layers
    if args.deepspeed and args.deepspeed_activation_checkpointing:
        set_deepspeed_activation_checkpointing(args, num_checkpoints)

    if dist.get_rank() == 0:
        print('Pretrain Enc-Dec model')
        print_args(args)
        with open(os.path.join(args.save, "args.json"), "w") as f:
            json.dump(vars(args), f)

    # Random seeds for reproducability.
    set_random_seed(args.seed)
    device = torch.cuda.current_device()

    # setup tokenizer
    tokenizer = EVATokenizer(os.path.join(args.tokenizer_path, 'vocab.txt'))
    config.vocab_size = tokenizer.vocab_size

    with open(args.deepspeed_config, "r") as f:
        ds_config = json.load(f)

    ds_config["gradient_accumulation_steps"] = args.gradient_accumulation_steps
    ds_config["train_micro_batch_size_per_gpu"] = args.batch_size

    if args.do_train:
        train_dataloader, train_dataset, _ = load_data(args, 'train', tokenizer, ratio=args.train_ratio)
        dev_dataloader, dev_dataset, _  = load_data(args, 'valid', tokenizer, ratio=args.valid_ratio)
        if args.train_iters == -1:
            args.train_iters = len(train_dataset) * args.epochs // (mpu.get_data_parallel_world_size() * args.batch_size * args.gradient_accumulation_steps)
        if args.save_interval == -1:
            args.save_interval = len(train_dataset) // (mpu.get_data_parallel_world_size() * args.batch_size * args.gradient_accumulation_steps)
        if args.eval_interval == -1:
            args.eval_interval = len(train_dataset) // (mpu.get_data_parallel_world_size() * args.batch_size * args.gradient_accumulation_steps)                
    else:
        args.train_iters = 10 # a magic number

    log_string = "Total train epochs {} | Total train iters {} | ".format(args.epochs, args.train_iters)
    print_rank_0(log_string)
    save_rank_0(args, log_string)

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, config, ds_config, args.do_train)
        
    if args.do_train:
        train(args, tokenizer, model, optimizer, lr_scheduler, train_dataset, train_dataloader, dev_dataset, dev_dataloader, device)

    if args.do_eval:
        eval_dataloader, eval_dataset, _ = load_data(args, 'test', tokenizer, ratio=args.test_ratio)
        loss, metrics, generation = evaluate(args, tokenizer, eval_dataset, eval_dataloader, model, device, mode="test")
        log_string = "Eval result: "
        for key, value in metrics.items():
            log_string += " {}: {:.5} | ".format(key, value)
        if dist.get_rank() == 0:
            with open(os.path.join(args.save, "metrics.json"), "w") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
            with open(os.path.join(args.save, "generation.json"), "w") as f:
                json.dump(generation, f, ensure_ascii=False, indent=2)
        print_rank_0(log_string)
        save_rank_0(args, log_string)

if __name__ == "__main__":
    main()
