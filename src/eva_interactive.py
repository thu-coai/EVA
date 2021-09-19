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

"""Redis Backend Generate EVA"""

USE_TORCH_DDP = False

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import time
from arguments import get_args
from utils import Timers
from utils import load_checkpoint
from tokenization_enc_dec import EncDecTokenizer
import mpu
import deepspeed
import torch.distributed as dist
from model import EncDecModel, EncDecConfig
from fp16 import FP16_Module
from utils import print_rank_0
from fuzzywuzzy import fuzz
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
from torch.nn.utils.rnn import pad_sequence

if USE_TORCH_DDP:
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
else:
    from model import DistributedDataParallel as DDP


class EncDecModelForInference(EncDecModel):
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        
    def forward(
        self,
        enc_input_ids=None,
        enc_position_ids=None,
        enc_attention_mask=None,
        dec_input_ids=None,
        dec_position_ids=None,
        dec_attention_mask=None,
        cross_attention_mask=None,
        enc_hidden_states=None,
        past_key_values=None,
    ):
        if enc_hidden_states is None:
            enc_outputs = self.encoder(
                input_ids=enc_input_ids,
                position_ids=enc_position_ids,
                attention_mask=enc_attention_mask,
            )
            return enc_outputs
        
        else:
            dec_outputs = self.decoder(
                input_ids=dec_input_ids,
                position_ids=dec_position_ids,
                attention_mask=dec_attention_mask,
                cross_attention_mask=cross_attention_mask,
                enc_hidden_states=enc_hidden_states,
                past_key_values=past_key_values,
            )
            last_hidden_state_parallel = mpu.copy_to_model_parallel_region(dec_outputs["last_hidden_state"])
            logits_parallel = F.linear(last_hidden_state_parallel, self.lm_head.weight)
    
            if self.parallel_output:
                lm_logits = logits_parallel
            else:
                lm_logits = mpu.gather_from_model_parallel_region(logits_parallel)
                
            return dec_outputs, lm_logits


def calc_banned_ngram_tokens(prev_input_ids, num_hypos: int, no_repeat_ngram_size: int, cur_len: int, vocab_size: int):
    generated_ngrams = [{tuple([23]):[33, 31], tuple([31]):[123]} for _ in range(num_hypos)]
    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        penalty_idx = tuple(prev_input_ids[hypo_idx, cur_len - 1: cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, []) + generated_ngrams[hypo_idx].get(penalty_idx, [])

    if cur_len + 1 < no_repeat_ngram_size:
        if cur_len > 0:
            return [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
        # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
        return [[] for _ in range(num_hypos)]
    #generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            if any(e >= vocab_size for e in ngram):
                continue
            prev_ngram_tuple = tuple(ngram[:-1])
            generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, []) + [ngram[-1]]

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


def get_model_for_inference(args, vocab_size):
    """Build the model."""

    print_rank_0('building Enc-Dec model ...')
    config = EncDecConfig.from_json_file(args.model_config)
    config.vocab_size = vocab_size
    assert not args.checkpoint_activations
    model = EncDecModelForInference(
        config,
        parallel_output=True,
        checkpoint_activations=False,
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

    # Wrap model for distributed training.
    if USE_TORCH_DDP:
        i = torch.cuda.current_device()
        model = DDP(model, device_ids=[i], output_device=i,
                    process_group=mpu.get_data_parallel_group())
    else:
        model = DDP(model)

    return model
    

def setup_model_for_inference(args, vocab_size):
    """Setup model and optimizer."""

    model = get_model_for_inference(args, vocab_size)

    if args.deepspeed:
        print_rank_0("DeepSpeed is enabled.")

        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=None,
            args=args,
            lr_scheduler=None,
            mpu=mpu,
            dist_init_required=False
        )

    if args.load is not None:
        args.iteration = load_checkpoint(model, optimizer, lr_scheduler, args)
    else:
        args.iteration = 0

    return model


def setup_ranker_for_inference(args,):
    """setup ranker model"""
    if args.ranker_config is None:
        return None, None
    ranker_tokenizer = BertTokenizer.from_pretrained(args.ranker_config)
    ranker_tokenizer.add_tokens(['<uttsep>'])
    ranker_tokenizer.add_special_tokens({"pad_token": '[PAD]'})
    
    ranker = BertForSequenceClassification.from_pretrained(args.ranker_config)
    ranker.resize_token_embeddings(len(ranker_tokenizer))
    param = torch.load(args.ranker_load)
    ranker.load_state_dict(param['state_dict'])
    ranker.eval()
    ranker.to(torch.cuda.current_device())

    return ranker, ranker_tokenizer


def get_masks_and_position_ids(args,
                               tokenizer,
                               contexts,
                               targets,
                               reset_position_ids,
                               reset_attention_mask):
    # Extract batch size and sequence length.
    batch_size, enc_seq_length = contexts.size()

    # Enc Attention mask.
    enc_attn_mask = torch.zeros(
        batch_size, 1, enc_seq_length, enc_seq_length, device=contexts.device)

    ctx_lengths = (contexts != tokenizer.pad_id).sum(1)
    for b in range(batch_size):
        enc_attn_mask[b, 0, :ctx_lengths[b], :ctx_lengths[b]] = 1

    # Enc Position ids.
    enc_pos_ids = torch.arange(
        enc_seq_length, dtype=torch.long, device=contexts.device)
    enc_pos_ids = enc_pos_ids.unsqueeze(0).expand_as(contexts)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        enc_pos_ids = enc_pos_ids.clone()

    batch_size, dec_seq_length = targets.size()
    # Dec Attention mask
    dec_attn_mask = torch.tril(torch.ones(
        batch_size, 1, dec_seq_length, dec_seq_length, device=targets.device))

    # Dec Position ids.
    dec_pos_ids = torch.arange(
        dec_seq_length, dtype=torch.long, device=targets.device)
    dec_pos_ids = dec_pos_ids.unsqueeze(0).expand_as(targets)
    # We need to clone as the ids will be modifed based on batch index.
    if reset_position_ids:
        dec_pos_ids = dec_pos_ids.clone()

    # Cross Attention Mask
    cross_attn_mask = torch.zeros(
        batch_size, 1, dec_seq_length, enc_seq_length, device=contexts.device)

    for b in range(batch_size):
        cross_attn_mask[b, 0, :, :ctx_lengths[b]] = 1

    if args.fp16:
        enc_attn_mask = enc_attn_mask.half()
        dec_attn_mask = dec_attn_mask.half()
        cross_attn_mask = cross_attn_mask.half()

    model_batch = {
        "enc_attention_mask": enc_attn_mask,
        "enc_position_ids": enc_pos_ids,
        "dec_attention_mask": dec_attn_mask,
        "dec_position_ids": dec_pos_ids,
        "cross_attention_mask": cross_attn_mask,
    }

    return model_batch


def get_inference_batch(
        context_tokens,
        device,
        batch_size,
        target_length,
        tokenizer,
        args,
    ):
    tokens = context_tokens
    tokens = tokens.view(batch_size, -1).contiguous()
    tokens = tokens.to(device)
    
    targets = torch.zeros(batch_size, target_length, dtype=torch.long, device=device) + tokenizer.get_sentinel_id(0)

    # Get the masks and postition ids.
    model_batch = get_masks_and_position_ids(
        args,
        tokenizer,
        tokens,
        targets,
        args.reset_position_ids,
        args.reset_attention_mask,
    )
    
    model_batch = {
        "enc_input_ids": tokens,
        "dec_input_ids": targets,
        **model_batch
    }

    return model_batch


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-10000, remove_unk=False):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if remove_unk:
        logits[..., 0] = filter_value

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    batch_size = logits.size()[0]
    if top_p > 0.0:
        logits=logits.view(batch_size, -1).contiguous()
        for logit in logits:
            sorted_logits, sorted_indices = torch.sort(logit, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logit[indices_to_remove] = filter_value

        logits=logits.view(batch_size, -1).contiguous()

    return logits

from human_rules import check_resp, get_resp, post_process, init_list


def rerank(model, tokenizer, device, context, responses):
    context = "<uttsep>".join(context)
    max_len = 128
    raw_batch = []
    for r in responses:
        item = tokenizer(context, r, max_length=max_len, truncation=True)
        item['input_ids'] = torch.tensor(item['input_ids'], dtype=torch.long) 
        item['attention_mask'] = torch.tensor(item['attention_mask'], dtype=torch.float)
        item['token_type_ids'] = torch.tensor(item['token_type_ids'], dtype=torch.long)
        raw_batch.append(item)
    batch = {}
    batch['input_ids'] = pad_sequence([x['input_ids'] for x in raw_batch], batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    batch['attention_mask'] = pad_sequence([x['attention_mask'] for x in raw_batch], batch_first=True, padding_value=0).to(device)
    batch['token_type_ids'] = pad_sequence([x['token_type_ids'] for x in raw_batch], batch_first=True, padding_value=0).to(device)
    outputs = model(**batch)
    logits = torch.softmax(outputs.logits, dim=-1)
    scores = logits[:, 1].tolist()
    select_id = np.argmax(scores)
    return select_id, scores
    
    


def generate_samples(model, tokenizer: EncDecTokenizer, args, device, ranker=None, ranker_tokenizer=None):
    init_list()
    no_repeat_ngram_size = 3
    repetition_penalty = 1.2
    batch_size = 1
    _min_sent_length = 15
    # _max_regenerate_times = 5 # 最大重复生成次数
    _sample_num = args.rerank_num
    _sep_p = 0.9
    again = False
    model.eval()
    if ranker is not None:
        ranker.eval()


    assert batch_size == 1
    batch_size *= _sample_num

    with torch.no_grad():
        
        all_input_tokens = []
        all_input_tokens_list = []
        context_utterances = []
        while True:
            min_sent_length = _min_sent_length # 最小句子长度
            sep_p = _sep_p # 结束概率
            # max_regenerate_times = _max_regenerate_times
            if dist.get_rank() == 0:
                input_text = input("Usr >>> ")
                if input_text == "clear":
                    print("Clear Dialog")
                    all_input_tokens = []
                    all_input_tokens_list = []
                    context_utterances = []
                    length_tensor = torch.tensor([-1], dtype=torch.long).to(device)
                elif input_text == "set min_length":
                    min_length_input = input("please enter the min_length: ")
                    _min_sent_length = float(min_length_input)
                    print("min_length set to", _min_sent_length)
                    all_input_tokens = []
                    all_input_tokens_list = []
                    length_tensor = torch.tensor([-1], dtype=torch.long).to(device)
                elif input_text == "set sep_p":
                    sep_p_input = input("please enter the sep_p: ")
                    _sep_p = float(sep_p_input)
                    print("sep_p set to", _sep_p)
                    all_input_tokens = []
                    all_input_tokens_list = []
                    length_tensor = torch.tensor([-1], dtype=torch.long).to(device)
                # elif input_text == "set again":
                #     again_input = input("please set again: (1 means true)")
                #     if(again_input == "1"):
                #         again = True
                #     else:
                #         again = False
                #     print("again is now ", again)
                #     all_input_tokens = []
                #     length_tensor = torch.tensor([-1], dtype=torch.long).to(device)
                else:
                    context_utterances.append(input_text)
                    resp = None
                    if args.human_rules:
                        resp = get_resp(all_input_tokens, input_text, tokenizer)
                    if resp is not None:
                        context_utterances.append(resp)
                        all_input_tokens_list.append(tokenizer.encode(input_text) + [tokenizer.sep_id] + tokenizer.encode(resp) + [tokenizer.sep_id])
                        length_tensor = torch.tensor([-1], dtype=torch.long).to(device)
                        print("Sys >>> ", resp)
                        # print(tokenizer.decode(all_input_tokens))
                    else:
                        all_input_tokens_list.append(tokenizer.encode(input_text) + [tokenizer.sep_id])
                        all_input_tokens = []
                        for utt in all_input_tokens_list[::-1]:
                            if len(all_input_tokens) + len(utt) + 1 <= 128:
                                all_input_tokens = utt + all_input_tokens
                        all_input_tokens.append(tokenizer.get_sentinel_id(0))
                        print(tokenizer.decode(all_input_tokens))
                        input_len = len(all_input_tokens)
                        length_tensor = torch.tensor([input_len], dtype=torch.long).to(device)
                        token_tensor = torch.tensor(all_input_tokens, dtype=torch.long).to(device)
            else:
                length_tensor = torch.zeros(1, dtype=torch.long).to(device)
            
            dist.barrier()
            dist.broadcast(length_tensor, 0)

            if length_tensor[0] < 0:
                continue
            if dist.get_rank() != 0:
                token_tensor = torch.zeros(int(length_tensor), dtype=torch.long).to(device)
            dist.broadcast(token_tensor, 0)
            token_tensor = token_tensor.unsqueeze(0).repeat(batch_size, 1) # repeat
            target_length = args.max_length
            model_batch = get_inference_batch(token_tensor, device, batch_size, target_length, tokenizer, args)

            min_sent_length = _min_sent_length # 最小句子长度
            sep_p = _sep_p # 结束概率
            
            enc_input_ids = model_batch['enc_input_ids']
            enc_attention_mask = model_batch['enc_attention_mask']
            enc_position_ids = model_batch['enc_position_ids']
            enc_outputs = model(
                enc_input_ids=enc_input_ids,
                enc_position_ids=enc_position_ids,
                enc_attention_mask=enc_attention_mask,
            )
            enc_hidden_states = enc_outputs["last_hidden_state"]
            
            # for generating responses
            # we only use the <go> token, so truncate other tokens
            dec_input_ids = model_batch['dec_input_ids'][..., :1]
            dec_attention_mask = model_batch['dec_attention_mask'][..., :1, :1]
            dec_position_ids = model_batch['dec_position_ids'][..., :1]
            # we use past_key_values, so only the current token mask is needed
            cross_attention_mask = model_batch['cross_attention_mask'][..., :1, :]
            
            unfinished_sents = enc_input_ids.new(enc_input_ids.size(0)).fill_(1)
            output_ids = enc_input_ids.new_zeros([enc_input_ids.size(0), 0])
            output_probs = torch.zeros(batch_size, 1).to(device)
            prob_idx = torch.arange(batch_size)
            past_key_values = None
            
            gen_len = 0
            # start_time = time.time()
            while gen_len < target_length:
                #print_rank_0(f'>>>>>> gen_len: {gen_len} <<<<<<')
                
                if unfinished_sents.max() == 0:
                    tokens_to_add = tokenizer.sep_id * (1 - unfinished_sents)
                    output_ids = torch.cat([output_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                
                else:
                    dec_outputs, lm_logits = model(
                        dec_input_ids=dec_input_ids,
                        dec_position_ids=dec_position_ids,
                        dec_attention_mask=dec_attention_mask,
                        cross_attention_mask=cross_attention_mask,
                        enc_hidden_states=enc_hidden_states,
                        past_key_values=past_key_values,
                    )
                    past_key_values = dec_outputs['past_key_values']
                    
                    gathered_lm_logits = [torch.zeros_like(lm_logits).to(device) for _ in range(mpu.get_model_parallel_world_size())]
                    torch.distributed.all_gather(gathered_lm_logits, lm_logits.data, mpu.get_model_parallel_group())
                    lm_logits = torch.cat(gathered_lm_logits, dim=-1)

                    logits = lm_logits[:, -1, :] / args.temperature

                    prev_output_tokens = torch.cat([enc_input_ids, output_ids], dim=-1)

                    # repetition_penalty
                    if repetition_penalty != 1.0:
                        for i in range(logits.size(0)):
                            for previous_token in set(prev_output_tokens[i].tolist()):
                                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                                if logits[i, previous_token] < 0:
                                    logits[i, previous_token] *= repetition_penalty
                                else:
                                    logits[i, previous_token] /= repetition_penalty

                    # no_repeat_ngram_size
                    if no_repeat_ngram_size > 0:
                        banned_batch_tokens = calc_banned_ngram_tokens(
                            output_ids, logits.size(0), no_repeat_ngram_size, gen_len, logits.size(1)
                        )
                        for i, banned_tokens in enumerate(banned_batch_tokens):
                            logits[i, banned_tokens] = -1e5

                    logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p, remove_unk=True)
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

                    # 设置最小句子长度
                    if(min_sent_length > 0):
                        for i in range(args.batch_size):
                            # if(next_token[i] == 4): 
                            if(next_token[i] == 4 and probs[i][4] < sep_p): 
                                logits[i][4] = -100000000
                                logits[i][0] = -100000000
                                logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
                                probs = F.softmax(logits, dim=-1)
                                next_token[i] = torch.multinomial(probs[i], num_samples=1)
                                # print_rank_0(next_token[i])

                    next_prob = probs[prob_idx, next_token]
                    tokens_to_add = next_token * unfinished_sents + tokenizer.sep_id * (1 - unfinished_sents)
                    probs_to_add = next_prob * unfinished_sents
                    output_probs = torch.cat([output_probs, probs_to_add.unsqueeze(-1)], dim=-1)
                    
                    dec_input_ids = tokens_to_add.unsqueeze(-1)
                    output_ids = torch.cat([output_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
                    dec_position_ids = dec_position_ids[:, -1:] + 1
                    # let the current token attend to all previous tokens
                    dec_attention_mask = torch.cat([dec_attention_mask, dec_attention_mask[:, :, :, -1:]], dim=-1)
                    
                gen_len += 1
                min_sent_length -= 1
                unfinished_sents.mul_(tokens_to_add.ne(tokenizer.sep_id).long())
            # if check_resp(generation_token_ids, tokenizer) and max_regenerate_times > 0:
            #     set_random_seed(random.randint(0, 1000))
            #     continue
            # else:
            #     break

            if dist.get_rank() == 0:
                output_ids = output_ids.cpu().tolist()
                generation_token_ids_list = []
                generation_str_list = []
                for e in output_ids:
                    generation_token_ids = e[:e.index(tokenizer.sep_id)] if tokenizer.sep_id in e else e
                    generation_token_ids = post_process(all_input_tokens, input_text, generation_token_ids, tokenizer)
                    if not check_resp(generation_token_ids, tokenizer): # pass test
                        generation_token_ids_list.append(generation_token_ids)
                        generation_str_list.append(tokenizer.decode(generation_token_ids))
                    #     print('[pass]: ', tokenizer.decode(generation_token_ids))
                    # else:
                    #     print('[fail]: ', tokenizer.decode(generation_token_ids))
                
                select_id = 0
                if ranker is not None:
                    select_id, scores = rerank(ranker, ranker_tokenizer, device, context_utterances, generation_str_list)
                
                    for response, score in zip(generation_str_list, scores):
                        print(f'response = {response}, score = {score}')

                generation_token_ids = generation_token_ids_list[select_id]
                # e = output_ids[0].cpu().tolist()
                # generation_token_ids = e[:e.index(tokenizer.sep_id)] if tokenizer.sep_id in e else e
                # generation_token_ids = post_process(all_input_tokens, input_text, generation_token_ids, tokenizer)
                all_input_tokens_list.append(generation_token_ids + [tokenizer.sep_id])
                context_utterances.append(generation_str_list[select_id])
                
                print("Sys >>> {}".format(tokenizer.decode(generation_token_ids)))
                # print(tokenizer.decode(all_input_tokens))


def initialize_distributed(args):
    """Initialize torch.distributed."""

    # Manually set the device ids.
    device = args.rank % torch.cuda.device_count()
    if args.local_rank is not None:
        device = args.local_rank
    torch.cuda.set_device(device)
    # Call the init process
    deepspeed.init_distributed()
    # Set the model-parallel / data-parallel communicators.
    mpu.initialize_model_parallel(args.model_parallel_size)


def set_random_seed(seed):
    """Set random seed for reproducability."""

    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        mpu.model_parallel_cuda_manual_seed(seed)


def main():
    """Main serving program."""

    print('Loading Model ...')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)


    # Ranker
    ranker, ranker_tokenizer = None, None
    if args.rerank:
        ranker, ranker_tokenizer = setup_ranker_for_inference(args)

    #get the tokenizer
    tokenizer = EncDecTokenizer(os.path.join(args.tokenizer_path, 'vocab.txt'))

    # Model, optimizer, and learning rate.
    model = setup_model_for_inference(args, tokenizer.vocab_size)

    # Timer.
    timers = Timers()

    #setting default batch size to 1
    args.batch_size = 1

    print('Model Loaded!')
    #generate samples
    generate_samples(model, tokenizer, args, torch.cuda.current_device(), ranker=ranker, ranker_tokenizer=ranker_tokenizer)
    

if __name__ == "__main__":
    main()



