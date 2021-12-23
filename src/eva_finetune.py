# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pretrain Enc-Dec"""

import os
import json
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler
import torch.distributed as dist

from arguments import get_args
from tokenization_enc_dec import EncDecTokenizer

import mpu
from utils import save_checkpoint, load_checkpoint
from utils import print_args, print_rank_0, save_rank_0
from utils import set_random_seed, initialize_distributed, set_deepspeed_activation_checkpointing
from model import EncDecModel, EncDecConfig, enc_dec_get_params_for_weight_decay_optimization
from samplers import DistributedBatchSampler, RandomSampler

from fp16 import FP16_Module, FP16_Optimizer
from learning_rates import AnnealingLR

import deepspeed
from apex.optimizers import FusedAdam as Adam

from generation_metrics import Metric

import signal
signal.signal(signal.SIGCHLD, signal.SIG_IGN)

from model import DistributedDataParallel as DDP


class EVADataset(Dataset):
    def __init__(self, args, tokenizer: EncDecTokenizer, path, split, ratio=1, num=-1, cache_path=None):
        super(EVADataset, self).__init__()

        self.args = args
        self.tokenizer = tokenizer
        self.max_enc_len = args.enc_seq_length
        self.max_dec_len = args.dec_seq_length
        self.pad_id = tokenizer.pad_id
        self.ratio = ratio

        if cache_path is None or not os.path.exists(os.path.join(cache_path, split + ".pkl")):
            print_rank_0("No cache, processing data")
            self.contexts, self.targets, self.labels = self.preprocess(path)
            if dist.get_rank() == 0:
                if cache_path is not None:
                    os.makedirs(cache_path, exist_ok=True)
                    with open(os.path.join(cache_path, split + ".pkl"), "wb") as f:
                        pickle.dump((self.contexts, self.targets, self.labels), f)
                else:
                    print_rank_0("Cache path is None, no cache saved")
            dist.barrier()
        else:
            with open(os.path.join(cache_path, split + ".pkl"), "rb") as f:
                print("Provide cache, loading pickle")
                self.contexts, self.targets, self.labels = pickle.load(f)

        if dist.get_rank() == 0:
            print_rank_0("Total Data Number: {}".format(len(self.contexts)))

    def preprocess(self, path):
        contexts = []
        targets = []
        labels = []

        with open(path, "r") as f:
            lines = f.readlines()
        
        for line in tqdm(lines[:int(self.ratio * len(lines))], desc="Loading data from {}".format(path), disable=(dist.get_rank() != 0)):
            line = line.strip().split("\t")
            line = [self.tokenizer.encode(utt) for utt in line]
            context = line[:-1]
            target = line[-1]

            trunc_context = []
            for c in context[::-1]:
                if len(c) + len(trunc_context) + 1 + 1 <= self.max_enc_len: # first 1 for <sep>, second 1 for <s_0>
                    trunc_context = c + [self.tokenizer.sep_id] + trunc_context
                else:
                    break
            if len(trunc_context) > 0 and len(target) <= self.max_dec_len:
                trunc_context = trunc_context + [self.tokenizer.get_sentinel_id(0)]
                target = [self.tokenizer.get_sentinel_id(0)] + target
                contexts.append(trunc_context)
                targets.append(target[:-1])
                labels.append(target[1:])
            else:
                continue

        return contexts, targets, labels

    def __getitem__(self, index):
        return (self.contexts[index], self.targets[index], self.labels[index])

    def __len__(self):
        return len(self.contexts)

    def collate(self, samples):
        bs = len(samples)
        contexts = [s[0] for s in samples]
        targets = [s[1] for s in samples]
        labels = [s[2] for s in samples]

        batch = {
            "enc_input_ids": torch.ones(bs, self.max_enc_len, dtype=torch.long) * self.pad_id,
            "dec_input_ids": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
            "enc_attention_mask": torch.zeros(bs, 1, self.max_enc_len, self.max_enc_len),
            "dec_attention_mask": torch.zeros(bs, 1, self.max_dec_len, self.max_dec_len),
            "cross_attention_mask": torch.zeros(bs, 1, self.max_dec_len, self.max_enc_len)
        }

        no_model_batch = {
            "labels": torch.ones(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
            "loss_mask": torch.zeros(bs, self.max_dec_len)
        }

        for b in range(bs):
            batch["enc_input_ids"][b, :len(contexts[b])] = torch.tensor(contexts[b], dtype=torch.long)
            batch["dec_input_ids"][b, :len(targets[b])] = torch.tensor(targets[b], dtype=torch.long)
            no_model_batch["labels"][b, :len(labels[b])] = torch.tensor(labels[b], dtype=torch.long)
            no_model_batch["loss_mask"][b, :len(labels[b])] = 1

            batch["enc_attention_mask"][b, 0, :len(contexts[b]), :len(contexts[b])] = 1
            batch["dec_attention_mask"][b, 0, :len(targets[b]), :len(targets[b])] = torch.tril(torch.ones(len(targets[b]), len(targets[b])))
            batch["cross_attention_mask"][b, 0, :len(targets[b]), :len(contexts[b])] = 1

        if self.args.fp16:
            batch["enc_attention_mask"] = batch["enc_attention_mask"].half()
            batch["dec_attention_mask"] = batch["dec_attention_mask"].half()
            batch["cross_attention_mask"] = batch["cross_attention_mask"].half()


        return batch, no_model_batch


def get_model(args, config):
    """Build the model."""

    print_rank_0('building Enc-Dec model ...')

    model = EncDecModel(
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

    if args.cpu_optimizer:
        if args.cpu_torch_adam:
            cpu_adam_optimizer = torch.optim.Adam
        else:
            from deepspeed.ops.adam import DeepSpeedCPUAdam
            cpu_adam_optimizer = DeepSpeedCPUAdam
        optimizer = cpu_adam_optimizer(param_groups,
                        lr=args.lr, weight_decay=args.weight_decay)
    else:
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

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
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


def setup_model_and_optimizer(args, model_config, ds_config):
    """Setup model and optimizer."""

    model = get_model(args, model_config)
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
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
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
    for k in model_batch:
        model_batch[k] = model_batch[k].to(device)
    for k in no_model_batch:
        no_model_batch[k] = no_model_batch[k].to(device)

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
                eval_loss, acc = evaluate(args, tokenizer, dev_dataset, dev_dataloader, model, device, mode="dev")
                model.train()
                log_string = prefix + " eval_loss: " + str(eval_loss)
                print_rank_0(log_string)
                save_rank_0(args, log_string)

            step += 1
            if step % args.gradient_accumulation_steps == 0:
                global_step += 1

    return global_step


def gen_metric(args, tokenizer: EncDecTokenizer, all_preds, all_labels):
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

    total_loss = 0.0
    step = 0

    with torch.no_grad():
        for model_batch, no_model_batch in eval_data_loader:
            forw_out = forward_step(args, model_batch, no_model_batch, model, device, keep_enc_hidden=True)
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            total_loss += loss

            step += 1

    total_loss /= step

    return total_loss, None


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()

    os.makedirs(args.save, exist_ok=True)
    config = EncDecConfig.from_json_file(args.model_config)

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
    tokenizer = EncDecTokenizer(os.path.join(args.tokenizer_path, 'vocab.txt'))
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
    else:
        args.train_iters = 10 # a magic number

    log_string = "Total train epochs {} | Total train iters {} | ".format(args.epochs, args.train_iters)
    print_rank_0(log_string)
    save_rank_0(args, log_string)

    # Model, optimizer, and learning rate.
    model, optimizer, lr_scheduler = setup_model_and_optimizer(args, config, ds_config)
        
    if args.do_train:
        train(args, tokenizer, model, optimizer, lr_scheduler, train_dataset, train_dataloader, dev_dataset, dev_dataloader, device)

    if args.do_eval:
        eval_dataloader, eval_dataset, _ = load_data(args, 'valid', tokenizer, ratio=args.test_ratio)

        loss, acc = evaluate(args, tokenizer, eval_dataset, eval_dataloader, model, device, mode="test")

        log_string = "Eval result: loss: {:.6} | acc(mrr): {}".format(loss, acc)
        print_rank_0(log_string)
        save_rank_0(args, log_string)

if __name__ == "__main__":
    main()
