# coding=utf-8

"""Finetune EVA"""

import os
import json
import torch

from torch.utils.data import DataLoader, SequentialSampler
from arguments import get_args
from model import EVATokenizer
from cross_entropy import vocab_parallel_cross_entropy

from utils import print_args, save_rank_0
from utils import set_random_seed
from model import EVAModel
from model import EVAConfig
from samplers import DistributedBatchSampler, RandomSampler

from generation_metrics import Metric
from generation_utils import generate_beam, generate_no_beam

from eva_datasets import EVADataset
from tqdm import tqdm

import signal
signal.signal(signal.SIGCHLD, signal.SIG_IGN)


def get_model(args, config):
    """Build the model."""

    print('building Enc-Dec model ...')
    model = EVAModel.from_pretrained(args.load, config=config)
    model.cuda(torch.cuda.current_device())
    return model

def load_data(args, data_type, tokenizer, ratio=1, drop_last=True):
    data_path = os.path.join(args.data_path, data_type + args.data_ext)

    # Data parallel arguments.
    world_size = 1
    rank = 0
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
    
    logits = output.logits
    forw_out = {
        "logits": logits
    }
    if keep_enc_hidden:
        forw_out["enc_hidden_states"] = enc_hidden_states
    losses = vocab_parallel_cross_entropy(logits.contiguous().float(),no_model_batch["labels"])

    loss_mask = no_model_batch["loss_mask"]
    losses = (losses * loss_mask).sum(-1) / loss_mask.sum(-1)
    loss = losses.mean()

    forw_out["loss"] = loss
    forw_out["loss_batch"] = losses
    
    return forw_out


def backward_step(args, loss, model, optimizer):
    # backward
    optimizer.zero_grad()
    loss.backward()

    # Update master gradients.
    if args.clip_grad > 0:
        if not args.fp16:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        else:
            optimizer.clip_master_grads(args.clip_grad)


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

                tmp_labels = [torch.zeros_like(no_model_batch["labels"]).to(device) for _ in range(1)]
                tmp_labels[0]  = no_model_batch["labels"].data

                tmp_output_ids = [torch.zeros_like(output_ids).to(device) for _ in range(1)]
                tmp_output_ids[0] = output_ids.data
                
                tmp_contexts = [torch.zeros_like(model_batch["enc_input_ids"]).to(device) for _ in range(1)]
                tmp_contexts[0] = model_batch["enc_input_ids"].data
                
                context_token_ids = sum([e.cpu().tolist() for e in tmp_contexts], [])
                context_token_ids = [e[:e.index(tokenizer.pad_id)] if tokenizer.pad_id in e else e for e in context_token_ids]
        
                label_token_ids = sum([e.cpu().tolist() for e in tmp_labels], [])
                label_token_ids = [e[:e.index(tokenizer.sep_id)] if tokenizer.sep_id in e else e for e in label_token_ids]

                generation_token_ids = sum([e.cpu().tolist() for e in tmp_output_ids], [])
                generation_token_ids = [e[:e.index(tokenizer.sep_id)] if tokenizer.sep_id in e else e for e in generation_token_ids]
                for lab, gen in zip(label_token_ids, generation_token_ids):
                    metric.forstr([tokenizer.decode(lab)], tokenizer.decode(gen))
                
                for ctx, lab, gen in zip(context_token_ids, label_token_ids, generation_token_ids):
                    generation_res.append({
                        'context': tokenizer.decode(ctx),
                        'response': tokenizer.decode(lab),
                        'generation': tokenizer.decode(gen),
                    })
                    if e == 0:
                        print(f'****** context: {tokenizer.decode(ctx)}\n'
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
    config.feed_forward_proj = 'gated-gelu'

    # Optional DeepSpeed Activation Checkpointing Features
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
    args.train_iters = 10 # a magic number

    log_string = "Total train epochs {} | Total train iters {} | ".format(args.epochs, args.train_iters)
    print(log_string)
    save_rank_0(args, log_string)

    # Model, optimizer, and learning rate.
    model = get_model(args, config)

    if args.do_eval:
        eval_dataloader, eval_dataset, _ = load_data(args, 'test', tokenizer, ratio=args.test_ratio)
        loss, metrics, generation = evaluate(args, tokenizer, eval_dataset, eval_dataloader, model, device, mode="test")
        log_string = "Eval result: "
        for key, value in metrics.items():
            log_string += " {}: {:.5} | ".format(key, value)

        with open(os.path.join(args.save, "metrics.json"), "w") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        with open(os.path.join(args.save, "generation.json"), "w") as f:
            json.dump(generation, f, ensure_ascii=False, indent=2)
        print(log_string)
        save_rank_0(args, log_string)

if __name__ == "__main__":
    main()
