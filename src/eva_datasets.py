# coding=utf-8

"""Datasets of EVA"""

import os
import pickle
import torch

from tqdm import tqdm
from torch.utils.data import Dataset
from model import EVATokenizer

def save_log(args,message):
    with open(args.log_file, "a") as f:
        f.write(message + "\n")
        f.flush()

class EVADataset(Dataset):
    def __init__(self, args, tokenizer: EVATokenizer, path, split, ratio=1, num=-1, cache_path=None):
        super(EVADataset, self).__init__()

        self.args = args
        self.tokenizer = tokenizer
        self.max_enc_len = args.enc_seq_length
        self.max_dec_len = args.dec_seq_length
        self.pad_id = tokenizer.pad_token_id
        self.ratio = ratio

        if cache_path is None or not os.path.exists(os.path.join(cache_path, split + ".pkl")):
            print("No cache, processing data")
            self.contexts, self.targets, self.labels = self.preprocess(path)
            if cache_path is not None:
                os.makedirs(cache_path, exist_ok=True)
                with open(os.path.join(cache_path, split + ".pkl"), "wb") as f:
                    pickle.dump((self.contexts, self.targets, self.labels), f)
            else:
                print("Cache path is None, no cache saved")
        else:
            with open(os.path.join(cache_path, split + ".pkl"), "rb") as f:
                print("Provide cache, loading pickle")
                self.contexts, self.targets, self.labels = pickle.load(f)

        print_str = "Path: {} | Ratio:{} | Max enc len: {} | Max dec len: {} | Data num: {}".format(path, ratio, self.max_enc_len, self.max_dec_len, len(self.contexts))
        print(print_str)
        save_log(args, print_str)

    def preprocess(self, path):
        contexts = []
        targets = []
        labels = []

        with open(path, "r") as f:
            lines = f.readlines()
        
        for line in tqdm(lines[:int(self.ratio * len(lines))], desc="Loading data from {}".format(path), disable=False):
            line = line.strip().split("\t")
            line = [self.tokenizer.encode(utt) for utt in line]
            if len(line) == 1:
                context = line[0]
                target = [0, 0] # empty dial
            else:
                context = line[:-1]
                target = line[-1]

            trunc_context = []
            for c in context[::-1]:
                if len(c) + len(trunc_context) + 1 + 1 <= self.max_enc_len: # first 1 for <sep>, second 1 for <s_0>
                    trunc_context = c + [self.tokenizer.eos_token_id] + trunc_context
                else:
                    break
            if len(trunc_context) > 0 and len(target) <= self.max_dec_len:
                trunc_context = trunc_context + [self.tokenizer.bos_token_id]
                label = [self.tokenizer.bos_token_id] + target + [self.tokenizer.eos_token_id]
                contexts.append(trunc_context)
                labels.append(label)
            else:
                continue

        return contexts, targets, labels

    def __getitem__(self, index):
        return (self.contexts[index], self.labels[index])

    def __len__(self):
        return len(self.contexts)

    def collate(self, samples):
        bs = len(samples)
        contexts = [s[0] for s in samples]
        labels = [s[1] for s in samples]

        batch = {
            "input_ids": torch.ones(bs, self.max_enc_len, dtype=torch.long) * self.pad_id,
            "attention_mask": torch.zeros(bs, self.max_enc_len, self.max_enc_len),
            "encoder_attention_mask": torch.zeros(bs, self.max_dec_len, self.max_dec_len), # cross attention mask
            "decoder_attention_mask": torch.zeros(bs, self.max_dec_len, self.max_enc_len),
            "labels": torch.zeros(bs, self.max_dec_len, dtype=torch.long) * self.pad_id,
        }

        for b in range(bs):
            context_length = len(contexts[b])
            label_length = len(labels[b])
            batch["input_ids"][b, :context_length] = torch.tensor(contexts[b], dtype=torch.long)
            batch["labels"][b, :label_length] = torch.tensor(labels[b], dtype=torch.long)

            batch["attention_mask"][b, :context_length, :context_length] = 1
            batch["decoder_attention_mask"][b, :label_length-1, :label_length-1] = torch.tril(torch.ones(label_length-1, label_length-1))
            batch["encoder_attention_mask"][b, :label_length-1, :context_length] = 1 # cross attention mask

        if self.args.fp16:
            batch["enc_attention_mask"] = batch["enc_attention_mask"].half()
            batch["dec_attention_mask"] = batch["dec_attention_mask"].half()
            batch["cross_attention_mask"] = batch["cross_attention_mask"].half()

        return batch