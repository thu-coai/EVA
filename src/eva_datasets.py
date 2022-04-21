# coding=utf-8

"""Datasets of EVA"""

import os
import pickle
import torch

import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data import Dataset
from tokenization_eva import EVATokenizer
from utils import print_rank_0, save_rank_0

class EVADataset(Dataset):
    def __init__(self, args, tokenizer: EVATokenizer, path, split, ratio=1, num=-1, cache_path=None):
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

        print_str = "Path: {} | Ratio:{} | Max enc len: {} | Max dec len: {} | Data num: {}".format(path, ratio, self.max_enc_len, self.max_dec_len, len(self.contexts))
        print_rank_0(print_str)
        save_rank_0(args, print_str)

    def preprocess(self, path):
        contexts = []
        targets = []
        labels = []

        with open(path, "r") as f:
            lines = f.readlines()
        
        for line in tqdm(lines[:int(self.ratio * len(lines))], desc="Loading data from {}".format(path), disable=(dist.get_rank() != 0)):
            line = line.strip().split("\t")
            line = [self.tokenizer.encode(utt) for utt in line]
            if len(line) == 1:
                context = line
                target = [0, 0] # empty dial
            else:
                context = line[:-1]
                target = line[-1]

            trunc_context = []
            for c in context[::-1]:
                if len(c) + len(trunc_context) + 1 + 1 <= self.max_enc_len: # first 1 for <sep>, second 1 for <s_0>
                    trunc_context = c + [self.tokenizer.sep_id] + trunc_context
                else:
                    break
            if len(trunc_context) > 0 and len(target) < self.max_dec_len:
                trunc_context = trunc_context + [self.tokenizer.get_sentinel_id(0)]
                target = [self.tokenizer.get_sentinel_id(0)] + target + [self.tokenizer.sep_id]
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
