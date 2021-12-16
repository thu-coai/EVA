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

from json import decoder
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import defaultdict
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


class BeamHypotheses(object):
    def __init__(self, num_beams, max_length, length_penalty, early_stopping, tokenizer=None):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.length_fact = []
        self.beams = []
        self.worst_score = 1e9
        self.raw_worst_score = 1e9

        self.tokenizer = tokenizer

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / len(hyp) ** self.length_penalty
        # print(f'add hyp = {self.tokenizer.decode(hyp.cpu().tolist())}, score = {score}')
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            self.length_fact.append(len(hyp) ** self.length_penalty)
            if len(self) > self.num_beams:
                sorted_scores = sorted([(s, idx, _) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_scores[0][1]]
                self.worst_score = sorted_scores[1][0]
                self.raw_worst_score = self.worst_score * (len(sorted_scores[1][2]) ** self.length_penalty)
            else:
                self.worst_score = min(score, self.worst_score)
                self.raw_worst_score = sum_logprobs
        
        # print('maintained hypothesis: ')
        # for score, hyp in self.beams:
        #     print(f'raw_score = {score * (len(hyp) ** self.length_penalty)}, score = {score}, hyp = {self.tokenizer.decode(hyp.cpu().tolist())}')

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            # print(f'cur best score = {cur_score}, cur worst score = {self.worst_score}, cur raw worst score = {self.raw_worst_score}')
            ret = self.worst_score >= cur_score
            return ret


def construct_antonym_dict(args):
    with open(os.path.join(args.rule_path, './antonym/antonym.txt'), 'r') as f:
        data = f.read().split("\n")
    data = [eval(item) for item in data if item]
    antonym_dict = defaultdict(list)

    for first, second in data:
        antonym_dict[first].append(second)
        antonym_dict[second].append(first)
    return antonym_dict


def calc_banned_antonym_words_ids(input_tokens, tokenizer, antonym_dict):
    antonym_words = [set()] * len(input_tokens)
    # only consider tokens occurring in current sentence
    for idx, tokens in enumerate(input_tokens):
        for word in tokenizer.convert_ids_to_tokens(reversed(tokens.tolist())):
            if word == '<sep>':
                break
            antonym_words[idx].update(tokenizer.convert_tokens_to_ids(antonym_dict[word]))

    return [list(tokens) for tokens in antonym_words]


def calc_banned_ngram_tokens(prev_input_ids, num_hypos: int, no_repeat_ngram_size: int, tokenizer: EncDecTokenizer) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
    # cur_len = prev_input_ids.size(-1)
    # # prev_input_words = tokenizer.decode(prev)
    # if cur_len + 1 < no_repeat_ngram_size:
    #     # return no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
    #     return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    prev_input_words = []
    for ids in prev_input_ids:
        tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
        words = []
        for token in tokens:
            if token == '<sep>':
                words.append(token)
            else:
                words += list(token)
        prev_input_words.append(words)
    for idx in range(num_hypos):
        gen_words = prev_input_words[idx]
        # print('gen_words = ', gen_words)
        # gen_tokens = prev_input_ids[idx].tolist()
        # gen_words = tokenizer.decode(gen_tokens)
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_words[i:] for i in range(no_repeat_ngram_size)]):
            for prefix_len in range(no_repeat_ngram_size):
                prev_ngram = ''.join(ngram[:prefix_len])
                suffix_ngram = ''.join(ngram[prefix_len:])
                if tokenizer.check(suffix_ngram): # 在词表中
                    generated_ngram[prev_ngram] = generated_ngram.get(prev_ngram, set()) | set([suffix_ngram])
            # prev_ngram_tuple = ''.join(ngram[:-1])
            # generated_ngram[prev_ngram_tuple] = generated_ngram.get(prev_ngram_tuple, set()) | set([ngram[-1]])
    
    # print('generated_ngrams = ', generated_ngrams)

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared

        cur_len = len(prev_input_words[hypo_idx])
        
        generated_ngram_idx = []
        '''
        3-gram, prefix的长度可以是2/1/0
        '''
        for prefix_len in range(no_repeat_ngram_size):
            # print('')
            ngram_words = ''.join(prev_input_words[hypo_idx][cur_len-prefix_len:])
            # print('prev_input = ', prev_input_words[hypo_idx])
            # print('ngram_words = ', ngram_words)
            generated_ngram_words = generated_ngrams[hypo_idx].get(ngram_words, [])
            # print('generated_ngram_words = ', generated_ngram_words)
            # print('all generated_ngrams = ', generated_ngrams[hypo_idx])
            generated_ngram_idx += tokenizer.convert_tokens_to_ids(generated_ngram_words)
            # generated_ngram_idx += [x for word in generated_ngram_words for x in tokenizer.get_prefix_id_list(word)]
            # print('generated_ngram_idx = ', generated_ngram_idx)
            # print('='*100)
        if prev_input_words[hypo_idx][-1] in ['，', ',']:
            generated_ngram_idx.append(tokenizer.convert_token_to_id('但'))
        return generated_ngram_idx

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
        decoder_token_tensor,
        device,
        batch_size,
        target_length,
        tokenizer,
        args,
    ):
    tokens = context_tokens
    tokens = tokens.view(batch_size, -1).contiguous()
    tokens = tokens.to(device)
    
    targets = torch.zeros(batch_size, 1, dtype=torch.long, device=device) + tokenizer.get_sentinel_id(0)
    targets = torch.cat([targets, decoder_token_tensor], dim=-1)

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


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-10000):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313

    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    batch_size = logits.size()[0]
    if top_p > 0.0:
        # logits : (batch_size, vocab_size)
        logits=logits.view(batch_size, -1).contiguous()
        # logits : (batch_size, vocab_size)
        for logit in logits:
            # logit: (vocab_size)
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

# from human_rules import check_resp, get_resp, post_process, init_list


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
    

def check_relative(model, tokenizer, device, context_utterance):
    context = context_utterance[:-1]
    usr_resp = context_utterance[-1]
    max_len = 128
    raw_batch, utt_index = [], []
    trunc_context = None
    for i in range(len(context) - 1, 0, -2):
        trunc_context = tokenizer("<uttsep>".join(context[i-1:]), usr_resp, max_length=max_len, truncation=True)
        raw_batch.append(tokenizer("<uttsep>".join(context[i-1:i+1]), usr_resp, max_length=max_len, truncation=True))
        utt_index.append(i)
        if len(trunc_context["input_ids"]) >= max_len:
            break
    if trunc_context is None:
        return None, []
    raw_batch.append(trunc_context)
    for item in raw_batch:
        item['input_ids'] = torch.tensor(item['input_ids'], dtype=torch.long) 
        item['attention_mask'] = torch.tensor(item['attention_mask'], dtype=torch.float)
        item['token_type_ids'] = torch.tensor(item['token_type_ids'], dtype=torch.long)

    batch = {}
    batch['input_ids'] = pad_sequence([x['input_ids'] for x in raw_batch], batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
    batch['attention_mask'] = pad_sequence([x['attention_mask'] for x in raw_batch], batch_first=True, padding_value=0).to(device)
    batch['token_type_ids'] = pad_sequence([x['token_type_ids'] for x in raw_batch], batch_first=True, padding_value=0).to(device)
    outputs = model(**batch)
    is_relative = torch.argmax(outputs.logits, dim=-1)
    is_relative = is_relative[:-1]
    relative_utt = [k for x, k in zip(is_relative, utt_index) if x == 1]
    return sorted(relative_utt), reversed(is_relative)


def calc_banned_bad_words_ids(prev_input_ids, bad_words_ids):
    banned_tokens = []

    def _tokens_match(prev_tokens, tokens):
        if len(tokens) == 0:
            # if bad word tokens is just one token always ban it
            return True
        if len(tokens) > len(prev_input_ids):
            # if bad word tokens are longer then prev input_ids they can't be equal
            return False

        if prev_tokens[-len(tokens) :] == tokens:
            # if tokens match
            return True
        else:
            return False

    for prev_input_ids_slice in prev_input_ids:
        banned_tokens_slice = []

        for banned_token_seq in bad_words_ids:
            assert len(banned_token_seq) > 0, "Banned words token sequences {} cannot have an empty list".format(
                bad_words_ids
            )

            if _tokens_match(prev_input_ids_slice.tolist(), banned_token_seq[:-1]) is False:
                # if tokens do not match continue
                continue

            banned_tokens_slice.append(banned_token_seq[-1])

        banned_tokens.append(banned_tokens_slice)

    return banned_tokens


def enforce_repetition_penalty_(lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty, tokenizer=None):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for i in range(batch_size * num_beams):
        for previous_token in prev_output_tokens[i].tolist():
            if previous_token == tokenizer.sep_id:
                continue
            # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
            if lprobs[i, previous_token] < 0:
                lprobs[i, previous_token] *= repetition_penalty
            else:
                lprobs[i, previous_token] /= repetition_penalty


def postprocess_next_token_scores(
    tokenizer: EncDecTokenizer,
    scores,
    input_ids,
    no_repeat_ngram_size,
    bad_words_ids,
    cur_len,
    min_length,
    max_length,
    eos_token_id,
    repetition_penalty,
    batch_size, 
    num_beams,
    antonym_dict,
):
    # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
    if repetition_penalty != 1.0:
        enforce_repetition_penalty_(
            scores, batch_size, num_beams, input_ids, repetition_penalty, tokenizer=tokenizer
        )

    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        scores[:, eos_token_id] = -10000

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(input_ids, num_batch_hypotheses, no_repeat_ngram_size, tokenizer=tokenizer)
        # from IPython import embed

        # embed()
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -10000

    if bad_words_ids is not None:
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

        for i, banned_tokens in enumerate(banned_tokens):
            scores[i, banned_tokens] = -10000
    
    # TODO：添加if条件
    # add antonym banned list
    banned_tokens = calc_banned_antonym_words_ids(input_ids, tokenizer, antonym_dict)

    for i, banned_tokens in enumerate(banned_tokens):
        scores[i, banned_tokens] = -10000

    scores[:, 0] = -50000

    return scores


def generate_no_beam(model_batch, token_tensor_full, model, tokenizer: EncDecTokenizer, args, device):
    batch_size = args.batch_size
    target_length = args.max_length
    
    dec_init_length = 1 # +1 for s_0
    
    enc_input_ids = model_batch['enc_input_ids']
    enc_attention_mask = model_batch['enc_attention_mask']
    enc_outputs = model(
        enc_input_ids=enc_input_ids,
        enc_attention_mask=enc_attention_mask,
    )
    enc_hidden_states = enc_outputs["last_hidden_state"]
    
    # for generating responses
    # we only use the <go> token, so truncate other tokens
    dec_input_ids = model_batch['dec_input_ids'][..., :dec_init_length]
    dec_attention_mask = model_batch['dec_attention_mask'][..., :dec_init_length, :dec_init_length]
    # we use past_key_values, so only the current token mask is needed
    cross_attention_mask = model_batch['cross_attention_mask'][..., :dec_init_length, :]
    
    unfinished_sents = enc_input_ids.new(enc_input_ids.size(0)).fill_(1)
    output_ids = enc_input_ids.new_zeros([enc_input_ids.size(0), 0]) # not include the prompt
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

            prev_output_tokens = torch.cat([token_tensor_full, output_ids], dim=-1)

            logits = postprocess_next_token_scores(
                tokenizer=tokenizer,
                scores=logits,
                input_ids=prev_output_tokens,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                bad_words_ids=[[0]],
                cur_len=gen_len,
                min_length=args.min_length,
                max_length=args.max_length,
                eos_token_id=tokenizer.sep_id,
                repetition_penalty=args.repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
                antonym_dict=None
            )

            logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            next_prob = probs[prob_idx, next_token]
            tokens_to_add = next_token * unfinished_sents + tokenizer.sep_id * (1 - unfinished_sents)
            probs_to_add = next_prob * unfinished_sents
            output_probs = torch.cat([output_probs, probs_to_add.unsqueeze(-1)], dim=-1)
            
            dec_input_ids = tokens_to_add.unsqueeze(-1)
            output_ids = torch.cat([output_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
            # let the current token attend to all previous tokens
            dec_attention_mask = torch.cat([dec_attention_mask[:, :, -1:, :], dec_attention_mask[:, :, -1:, -1:]], dim=-1)
            cross_attention_mask = cross_attention_mask[:, :, -1:, :]

        gen_len += 1
        unfinished_sents.mul_(tokens_to_add.ne(tokenizer.sep_id).long())
    
    output_ids = output_ids.cpu().tolist()
    generation_token_ids_list = []
    generation_str_list = []
    for e in output_ids:
        generation_token_ids = e[:e.index(tokenizer.sep_id)] if tokenizer.sep_id in e else e
        generation_token_ids_list.append(generation_token_ids)
        generation_str_list.append(tokenizer.decode(generation_token_ids))
    
    return generation_str_list, generation_token_ids_list


def generate_beam(model_batch, token_tensor_full, model, tokenizer: EncDecTokenizer, args, device):
    batch_size = args.batch_size
    num_beams = args.num_beams
    target_length = args.max_length
    
    do_sample = args.top_p > 0 or args.top_k > 0
    vocab_size = tokenizer.vocab_size
    
    enc_input_ids = model_batch['enc_input_ids']
    enc_attention_mask = model_batch['enc_attention_mask']
    
    enc_input_length = enc_input_ids.size(-1)
    enc_input_ids = enc_input_ids.unsqueeze(1).expand(batch_size, num_beams, enc_input_length)
    enc_attention_mask = enc_attention_mask.unsqueeze(1).expand(batch_size, num_beams, 1, enc_input_length, enc_input_length)
    
    enc_input_ids = enc_input_ids.contiguous().view(batch_size * num_beams, enc_input_length)
    enc_attention_mask = enc_attention_mask.contiguous().view(batch_size * num_beams, 1, enc_input_length, enc_input_length)
    
    token_tensor_full = token_tensor_full.unsqueeze(1).expand(batch_size, num_beams, token_tensor_full.size(-1))
    token_tensor_full = token_tensor_full.contiguous().view(batch_size * num_beams, token_tensor_full.size(-1))
    
    enc_outputs = model(
        enc_input_ids=enc_input_ids,
        enc_attention_mask=enc_attention_mask,
    )
    enc_hidden_states = enc_outputs["last_hidden_state"]

    dec_init_length = 1 # +1 for s_0
    # for generating responses
    # we only use the <go> token, so truncate other tokens
    dec_input_ids = model_batch['dec_input_ids'][..., :dec_init_length]
    dec_attention_mask = model_batch['dec_attention_mask'][..., :dec_init_length, :dec_init_length]
    # we use past_key_values, so only the current token mask is needed
    cross_attention_mask = model_batch['cross_attention_mask'][..., :dec_init_length, :]
    
    dec_input_ids = dec_input_ids.unsqueeze(1).expand(batch_size, num_beams, dec_init_length)
    dec_attention_mask = dec_attention_mask.unsqueeze(1).expand(batch_size, num_beams, 1, dec_init_length, dec_init_length)
    cross_attention_mask = cross_attention_mask.unsqueeze(1).expand(batch_size, num_beams, 1, dec_init_length, enc_input_length)
    
    dec_input_ids = dec_input_ids.contiguous().view(batch_size * num_beams, dec_init_length)
    dec_attention_mask = dec_attention_mask.contiguous().view(batch_size * num_beams, 1, dec_init_length, dec_init_length)
    cross_attention_mask = cross_attention_mask.contiguous().view(batch_size * num_beams, 1, dec_init_length, enc_input_length)
    
    done = [False for _ in range(batch_size)]
    output_ids = enc_input_ids.new_zeros([enc_input_ids.size(0), 0]) # not include the prompt
    past_key_values = None
    
    gen_len = 0

    # construct antonym dict
    antonym_dict = construct_antonym_dict(args)

    # generated hypotheses
    generated_hyps = [
        BeamHypotheses(num_beams, target_length, args.length_penalty, early_stopping=args.early_stopping, tokenizer=tokenizer)
        for _ in range(batch_size)
    ]

    beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=dec_input_ids.device)
    beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

    while gen_len < target_length:
        dec_outputs, lm_logits = model(
            dec_input_ids=dec_input_ids,
            dec_attention_mask=dec_attention_mask,
            cross_attention_mask=cross_attention_mask,
            enc_hidden_states=enc_hidden_states,
            past_key_values=past_key_values,
        )
        past_key_values = dec_outputs['past_key_values']

        logits = lm_logits[:, -1, :] / args.temperature
        scores = F.log_softmax(logits, dim=-1)

        prev_output_tokens = torch.cat([token_tensor_full, output_ids], dim=-1)

        scores = postprocess_next_token_scores(
            tokenizer=tokenizer,
            scores=scores,
            input_ids=prev_output_tokens,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            bad_words_ids=None,
            cur_len=gen_len,
            min_length=args.min_length,
            max_length=args.max_length,
            eos_token_id=tokenizer.sep_id,
            repetition_penalty=args.repetition_penalty,
            batch_size=batch_size,
            num_beams=num_beams,
            antonym_dict=antonym_dict
        )

        if do_sample:
            _scores = scores + beam_scores[:, None].expand_as(scores)
            if args.temperature != 1.0:
                _scores = _scores / args.temperature                
            _scores = top_k_logits(_scores, top_k=args.top_k, top_p=args.top_p)
            _scores = _scores.contiguous().view(batch_size, num_beams * vocab_size)
            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
            probs = F.softmax(_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
            # Compute next scores
            next_scores = torch.gather(_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
            # sort the sampled vector to make sure that the first num_beams samples are the best
            next_scores, next_scores_indices = torch.sort(next_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)            
        else:
            next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

            # re-organize to group the beam together (we are keeping top hypothesis accross beams)
            next_scores = next_scores.view(
                batch_size, num_beams * vocab_size
            )  # (batch_size, num_beams * vocab_size)

            next_scores, next_tokens = torch.topk(next_scores, 2 * num_beams, dim=1, largest=True, sorted=True)

        assert next_scores.size() == next_tokens.size() == (batch_size, 2 * num_beams)
        # next batch beam content
        next_batch_beam = []

        for batch_idx in range(batch_size):
            # if we are done with this sentence, add a pad token
            if done[batch_idx]:
                assert (
                    len(generated_hyps[batch_idx]) >= num_beams
                ), "Batch can only be done if at least {} beams have been generated".format(num_beams)
                next_batch_beam.extend([(0, tokenizer.pad_id, 0)] * num_beams)  # pad the batch
                continue

            # next sentence beam content, this will get added to next_batch_beam
            next_sent_beam = []

            # next tokens for this sentence
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(
                zip(next_tokens[batch_idx], next_scores[batch_idx])
            ):
                # get beam and token IDs
                beam_id = beam_token_id // vocab_size
                token_id = beam_token_id % vocab_size

                effective_beam_id = batch_idx * num_beams + beam_id
                # add to generated hypotheses if end of sentence
                if token_id.item() == tokenizer.sep_id:
                    # if beam_token does not belong to top num_beams tokens, it should not be added
                    is_beam_token_worse_than_top_num_beams = beam_token_rank >= num_beams
                    if is_beam_token_worse_than_top_num_beams:
                        continue
                    generated_hyps[batch_idx].add(
                        output_ids[effective_beam_id].clone(), beam_token_score.item(),
                    )
                else:
                    # add next predicted token since it is not eos_token
                    next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                # once the beam for next step is full, don't add more tokens to it.
                if len(next_sent_beam) == num_beams:
                    break

            # Check if we are done so that we can save a pad step if all(done)
            # is_done: the best candiates in the current beam is worse than the sentences already in generated_hyps
            # print('cur worst score = ', generated_hyps[batch_idx].worst_score)
            # print('cur raw worst score = ', generated_hyps[batch_idx].raw_worst_score)
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done( # TODO: length penalty could influence the ending of generation
                next_scores[batch_idx].max().item(), gen_len
            )
            # for score, token_id, effective_beam_id in next_sent_beam:
            #     print(f'raw_socre = {score}, score = {score / gen_len ** args.length_penalty}, sentence = {tokenizer.decode(torch.cat([output_ids[effective_beam_id], token_id.unsqueeze(dim=0)], dim=-1).cpu().tolist())}')
            # print(f'id_done = {done[batch_idx]}')
            # print('='*100)

            # update next beam content
            assert len(next_sent_beam) == num_beams, "Beam should always be full"
            next_batch_beam.extend(next_sent_beam)
            assert len(next_batch_beam) == num_beams * (batch_idx + 1), "We should have added num_beams each step"

        # stop when we are done with each sentence
        if all(done):
            break

        # sanity check / prepare next batch
        assert len(next_batch_beam) == batch_size * num_beams
        beam_scores = torch.tensor([x[0] for x in next_batch_beam], device=dec_input_ids.device)
        beam_tokens = torch.tensor([x[1] for x in next_batch_beam], device=dec_input_ids.device)
        beam_idx = torch.tensor([x[2] for x in next_batch_beam], device=dec_input_ids.device)

        # re-order batch and update current length
        output_ids = output_ids[beam_idx, :]
        output_ids = torch.cat([output_ids, beam_tokens.unsqueeze(1)], dim=-1)

        dec_input_ids = beam_tokens.unsqueeze(1)
        dec_attention_mask = torch.cat([dec_attention_mask[:, :, -1:, :], dec_attention_mask[:, :, -1:, -1:]], dim=-1)
        cross_attention_mask = cross_attention_mask[:, :, -1:, :]

        past_key_values = [[torch.index_select(layer_past_type, 0, beam_idx) for layer_past_type in layer_past] for layer_past in past_key_values]
        
        gen_len += 1

    # finalize all open beam hypotheses and add to generated hypotheses
    for batch_idx in range(batch_size):
        if done[batch_idx]:
            continue

        # need to add best num_beams hypotheses to generated hyps
        for beam_id in range(num_beams):
            effective_beam_id = batch_idx * num_beams + beam_id
            final_score = beam_scores[effective_beam_id].item()
            final_tokens = output_ids[effective_beam_id]
            generated_hyps[batch_idx].add(final_tokens, final_score)

    best = []
    best_ids = []

    # retrieve best hypotheses
    for i, hypotheses in enumerate(generated_hyps):
        sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
        # for score, hyp in sorted_hyps:
        #     print(f'score = {score}, hyp = {tokenizer.decode(hyp.cpu().tolist())}')
        best_hyp = sorted_hyps.pop()[1]
        best.append(tokenizer.decode(best_hyp.cpu().tolist()))
        best_ids.append(best_hyp.cpu().tolist())

    return best, best_ids


def generate_samples(model, tokenizer: EncDecTokenizer, args, device, ranker=None, ranker_tokenizer=None):
    # init_list()
    model.eval()
    if ranker is not None:
        ranker.eval()

    with torch.no_grad():
        all_input_tokens_list = []
        context_utterances = []
        while True:
            if dist.get_rank() == 0:
                input_text = input("Usr >>> ")
                if input_text == "clear":
                    print("Clear Dialog")
                    set_random_seed(args.seed) # reset rng
                    all_input_tokens_list = []
                    context_utterances = []
                    length_tensor = torch.tensor([-1], dtype=torch.long).to(device)
                elif input_text.startswith("set len penalty:"):
                    args.length_penalty = float(input_text.split(":")[-1].strip())
                    print(f"set length penalty to {args.length_penalty}, Clear Dialog")
                    set_random_seed(args.seed) # reset rng
                    all_input_tokens_list = []
                    context_utterances = []
                    length_tensor = torch.tensor([-1], dtype=torch.long).to(device)
                elif input_text.startswith("set repetition penalty:"):
                    args.repetition_penalty = float(input_text.split(":")[-1].strip())
                    print(f"set repetition penalty to {args.repetition_penalty}, Clear Dialog")
                    set_random_seed(args.seed) # reset rng
                    all_input_tokens_list = []
                    context_utterances = []
                    length_tensor = torch.tensor([-1], dtype=torch.long).to(device)
                else:
                    context_utterances.append(input_text)
                    all_input_tokens_list.append(tokenizer.encode(input_text) + [tokenizer.sep_id])
                    resp = None
                    # if args.human_rules:
                    #     resp = get_resp(context_utterances, input_text, tokenizer)
                    if resp is not None and not resp["continue"]:
                        # print("resp in repo", resp)
                        resp = resp["resp"]
                        context_utterances.append(resp)
                        all_input_tokens_list.append(tokenizer.encode(resp) + [tokenizer.sep_id])
                        length_tensor = torch.tensor([-1], dtype=torch.long).to(device)
                        print("Sys >>> ", resp)
                        # print(tokenizer.decode(all_input_tokens))
                    else:
                        prompt = ""
                        if resp is not None:
                            prompt = resp["resp"]
                        if ranker is not None and ranker_tokenizer is not None:
                            trunc_index, is_relative = check_relative(ranker, ranker_tokenizer, device, context_utterances)
                        else:
                            trunc_index = None
                        # print("trunc_index", trunc_index, "is_relative", is_relative)
                        all_input_tokens_full = [x for y in all_input_tokens_list for x in y]
                        trunc_list = all_input_tokens_list
                        if trunc_index is not None:
                            trunc_list = []
                            for k in trunc_index:
                                trunc_list.extend(all_input_tokens_list[k-1:k+1])
                            trunc_list.append(all_input_tokens_list[-1])
                        # print("trunc_k", trunc_k, "trunc_scores", trunc_scores)
                        # trunk_list = all_input_tokens_list[trunc_k:]
                        # all_input_tokens = []
                        # for utt in all_input_tokens_list[::-1]:
                        all_input_tokens = []
                        for utt in trunc_list[:-9:-1]:
                            if len(all_input_tokens) + len(utt) + 1 <= 128:
                                all_input_tokens = utt + all_input_tokens
                        all_input_tokens.append(tokenizer.get_sentinel_id(0))
                        # print(tokenizer.decode(all_input_tokens))
                        prompt_tokens = tokenizer.encode(prompt)                        
                        input_len = len(all_input_tokens)
                        prompt_len = len(prompt_tokens)
                        length_tensor = torch.tensor([input_len, prompt_len], dtype=torch.long).to(device)
                        token_tensor = torch.tensor(all_input_tokens, dtype=torch.long).to(device)
                        token_tensor_full = torch.tensor(all_input_tokens_full, dtype=torch.long).to(device)
                        decoder_token_tensor = torch.tensor(prompt_tokens, dtype=torch.long).to(device)

            else:
                length_tensor = torch.zeros(2, dtype=torch.long).to(device)
            
            # encoder tensor
            dist.barrier()
            dist.broadcast(length_tensor, 0)
            if length_tensor[0] < 0:
                continue
            if dist.get_rank() != 0:
                token_tensor = torch.zeros(int(length_tensor[0]), dtype=torch.long).to(device)
            dist.broadcast(token_tensor, 0)
            if dist.get_rank() != 0:
                decoder_token_tensor = torch.zeros(int(length_tensor[1]), dtype=torch.long).to(device)
            if length_tensor[1] > 0:
                dist.broadcast(decoder_token_tensor, 0)

            token_tensor = token_tensor.unsqueeze(0).repeat(args.batch_size, 1) # repeat
            token_tensor_full = token_tensor_full.unsqueeze(0).repeat(args.batch_size, 1) 
            decoder_token_tensor = decoder_token_tensor.unsqueeze(0).repeat(args.batch_size, 1)
            target_length = args.max_length
            model_batch = get_inference_batch(token_tensor, decoder_token_tensor, device, args.batch_size, target_length, tokenizer, args)
   
            if args.num_beams == 1:
                generation_str_list, generation_id_list = generate_no_beam(model_batch, token_tensor_full, model, tokenizer, args, device)
            else:
                generation_str_list, generation_id_list  = generate_beam(model_batch, token_tensor_full, model, tokenizer, args, device)

            all_input_tokens_list.append(generation_id_list[0] + [tokenizer.sep_id])
            context_utterances.append(generation_str_list[0])

            if dist.get_rank() == 0:
                print("Sys >>> {}".format(generation_str_list[0]))


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
    # os.system('clear')
    print('Model Loaded!')
    print("enter")
    #generate samples
    generate_samples(model, tokenizer, args, torch.cuda.current_device(), ranker=ranker, ranker_tokenizer=ranker_tokenizer)
    

if __name__ == "__main__":
    main()



