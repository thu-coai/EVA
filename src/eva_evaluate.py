# coding=utf-8

"""Evaluate EVA"""

import os
import json
import torch

from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from arguments import get_args
from model import EVATokenizer

from utils import print_args, save_rank_0
from utils import set_random_seed
from model import EVAModel

from generation_metrics import Metric

from eva_datasets import EVADataset
from tqdm import tqdm


def load_data(args, data_type, tokenizer, ratio=1, drop_last=True):
    data_path = os.path.join(args.data_path, data_type + args.data_ext)

    # Data parallel arguments.
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size

    global_batch_size = args.eval_batch_size * world_size

    num_workers = args.num_workers

    dataset = EVADataset(
        args,
        tokenizer,
        data_path,
        data_type,
        ratio=ratio,
        cache_path=args.cache_path)

    batch_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=drop_last)

    data_loader = DataLoader(dataset,
                             batch_size=args.eval_batch_size,
                             num_workers=num_workers,
                             pin_memory=True,
                             shuffle=False,
                             drop_last=drop_last,
                             collate_fn=dataset.collate)

    # Torch dataloader.
    return data_loader, dataset, batch_sampler


def evaluate(args, tokenizer, eval_dataset, eval_data_loader, model:EVAModel, device, mode="dev"):
    """Evaluation."""

    model.eval()

    metric = Metric(tokenizer)

    loss_res = None
    generation_res = None
    metric_res = {}
    
    step = 0

    with torch.no_grad():
        loss_res = 0.0
        for batch in tqdm(eval_data_loader, desc="Computing Loss"):
            for k in batch:
                batch[k] = batch[k].to(device)
            forw_out = model(**batch)
            loss = forw_out["loss"].item() if "loss" in forw_out else 0
            loss_res += loss
            step += 1

        loss_res /= step
        
        print(loss_res)

        if args.eval_generation:
            generation_res = []
            for e, batch in enumerate(tqdm(eval_data_loader, desc="Evaluating")):
                model_gen_tokens = model.generate(
                    batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    max_length=args.max_generation_length,
                    min_length=args.min_generation_length,
                    do_sample=args.do_sample,
                    num_beams=args.num_beams,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    length_penalty=args.length_penalty,
                    no_repeat_ngram_size=args.no_repeat_ngram_size,
                    encoder_no_repeat_ngram_size=args.no_repeat_ngram_size,
                    repetition_penalty=args.repetition_penalty,
                    use_cache=True
                )
                model_gen_str = tokenizer.batch_decode(model_gen_tokens, skip_special_tokens=True)
                label_str = tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
                context_str = tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True)

                for lab, gen in zip(label_str, model_gen_str):
                    metric.forstr([lab], gen)
                
                for ctx, lab, gen in zip(context_str, label_str, model_gen_str):
                    generation_res.append({
                        'context': ctx,
                        'response': lab,
                        'generation': gen,
                    })
                    if e == 0:
                        print(f'****** context: {ctx}\n'
                                    f'****** response: {lab}\n'
                                    f'****** generation: {gen}\n')

            metric_res, *_ = metric.close()

        metric_res["loss"] = loss_res

    return loss_res, metric_res, generation_res


def main():
    """Main training program."""

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()

    dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)

    os.makedirs(args.save, exist_ok=True)

    print("Evaluate EVA model")
    print_args(args)
    with open(os.path.join(args.save, "args.json"), "w") as f:
        json.dump(vars(args), f)

    set_random_seed(args.seed)
    device = torch.cuda.current_device()

    tokenizer = EVATokenizer.from_pretrained(args.load)
    model = EVAModel.from_pretrained(args.load)
    model = model.half().to(device)
    
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
