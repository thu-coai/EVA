# coding=utf-8

import os
import torch
import mpu

import torch.nn.functional as F

from collections import defaultdict
from tokenization_eva import EVATokenizer


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
            ret = self.worst_score >= cur_score
            return ret


def construct_antonym_dict(args):
    if args.rule_path is None:
        return None
    with open(os.path.join(args.rule_path, './antonym/antonym.txt'), 'r', encoding="utf-8") as f:
        data = f.read().split("\n")
    data = [eval(item) for item in data if item]
    antonym_dict = defaultdict(list)

    for first, second in data:
        antonym_dict[first].append(second)
        antonym_dict[second].append(first)
    return antonym_dict


def calc_banned_antonym_words_ids(input_tokens, tokenizer, antonym_dict):
    if antonym_dict is None:
        return []
    antonym_words = [set()] * len(input_tokens)
    # only consider tokens occurring in current sentence
    for idx, tokens in enumerate(input_tokens):
        for word in tokenizer.convert_ids_to_tokens(reversed(tokens.tolist())):
            if word == '<sep>':
                break
            antonym_words[idx].update(tokenizer.convert_tokens_to_ids(antonym_dict[word]))

    return [list(tokens) for tokens in antonym_words]


def calc_banned_ngram_tokens(prev_input_ids, num_hypos: int, no_repeat_ngram_size: int, tokenizer: EVATokenizer) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""
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
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(*[gen_words[i:] for i in range(no_repeat_ngram_size)]):
            for prefix_len in range(no_repeat_ngram_size):
                prev_ngram = ''.join(ngram[:prefix_len])
                suffix_ngram = ''.join(ngram[prefix_len:])
                if tokenizer.check(suffix_ngram): # 在词表中
                    generated_ngram[prev_ngram] = generated_ngram.get(prev_ngram, set()) | set([suffix_ngram])

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have already appeared

        cur_len = len(prev_input_words[hypo_idx])
        
        generated_ngram_idx = []
        '''
        3-gram, prefix的长度可以是2/1/0
        '''
        for prefix_len in range(no_repeat_ngram_size):
            ngram_words = ''.join(prev_input_words[hypo_idx][cur_len-prefix_len:])
            generated_ngram_words = generated_ngrams[hypo_idx].get(ngram_words, [])
            generated_ngram_idx += tokenizer.convert_tokens_to_ids(generated_ngram_words)
        if prev_input_words[hypo_idx][-1] in ['，', ',', '。']:
            generated_ngram_idx.append(tokenizer.convert_token_to_id('但'))
            generated_ngram_idx.append(tokenizer.convert_token_to_id('不过'))
        return generated_ngram_idx

    banned_tokens = [_get_generated_ngrams(hypo_idx) for hypo_idx in range(num_hypos)]
    return banned_tokens


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


def enforce_repetition_penalty_(tokenizer, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
    """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
    for i in range(batch_size * num_beams):
        for previous_token in set(prev_output_tokens[i].tolist()):
            if previous_token != tokenizer.sep_id:
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty


def postprocess_next_token_scores(
    tokenizer: EVATokenizer,
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
            tokenizer, scores, batch_size, num_beams, input_ids, repetition_penalty,
        )

    # set eos token prob to zero if min_length is not reached
    if eos_token_id is not None and cur_len < min_length:
        scores[:, eos_token_id] = -10000

    if no_repeat_ngram_size > 0:
        # calculate a list of banned tokens to prevent repetitively generating the same ngrams
        num_batch_hypotheses = batch_size * num_beams
        # from fairseq: https://github.com/pytorch/fairseq/blob/a07cb6f40480928c9e0548b737aadd36ee66ac76/fairseq/sequence_generator.py#L345
        banned_batch_tokens = calc_banned_ngram_tokens(input_ids, num_batch_hypotheses, no_repeat_ngram_size, tokenizer=tokenizer)
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores[i, banned_tokens] = -10000

    if bad_words_ids is not None:
        # calculate a list of banned tokens according to bad words
        banned_tokens = calc_banned_bad_words_ids(input_ids, bad_words_ids)

        for i, banned_tokens in enumerate(banned_tokens):
            scores[i, banned_tokens] = -10000
    
    # add antonym banned list
    banned_tokens = calc_banned_antonym_words_ids(input_ids, tokenizer, antonym_dict)

    for i, banned_tokens in enumerate(banned_tokens):
        scores[i, banned_tokens] = -10000

    scores[:, 0] = -50000

    return scores


def generate_no_beam(model_batch, full_context, model, tokenizer: EVATokenizer, args, device):
    target_length = args.max_generation_length
    
    dec_init_length = 1 # +1 for s_0
    
    enc_input_ids = model_batch['enc_input_ids']
    enc_attention_mask = model_batch['enc_attention_mask']
    enc_outputs = model(
        enc_input_ids=enc_input_ids,
        enc_attention_mask=enc_attention_mask,
        only_encoder=True
    )
    enc_hidden_states = enc_outputs["encoder_last_hidden_state"]
    
    batch_size = enc_input_ids.size(0)

    # for generating responses
    # we only use the <go> token, so truncate other tokens
    dec_input_ids = model_batch['dec_input_ids'][..., :dec_init_length]
    dec_attention_mask = model_batch['dec_attention_mask'][..., :dec_init_length, :dec_init_length]
    # we use past_key_values, so only the current token mask is needed
    cross_attention_mask = model_batch['cross_attention_mask'][..., :dec_init_length, :]
    
    unfinished_sents = enc_input_ids.new(enc_input_ids.size(0)).fill_(1)
    output_ids = enc_input_ids.new_zeros([enc_input_ids.size(0), 0]) # not include the prompt
    prob_idx = torch.arange(batch_size)
    past_key_values = None
    
    gen_len = 0
    # construct antonym dict
    antonym_dict = construct_antonym_dict(args)
    while gen_len < target_length:
        if unfinished_sents.max() == 0:
            tokens_to_add = tokenizer.sep_id * (1 - unfinished_sents)
            output_ids = torch.cat([output_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        
        else:
            dec_outputs = model(
                dec_input_ids=dec_input_ids,
                dec_attention_mask=dec_attention_mask,
                cross_attention_mask=cross_attention_mask,
                enc_hidden_states=enc_hidden_states,
                past_key_values=past_key_values,
            )
            past_key_values = dec_outputs['past_key_values']
            lm_logits = dec_outputs['lm_logits']
            
            gathered_lm_logits = [torch.zeros_like(lm_logits).to(device) for _ in range(mpu.get_model_parallel_world_size())]
            torch.distributed.all_gather(gathered_lm_logits, lm_logits.data, mpu.get_model_parallel_group())
            lm_logits = torch.cat(gathered_lm_logits, dim=-1)

            logits = lm_logits[:, -1, :] / args.temperature

            prev_output_tokens = torch.cat([full_context, output_ids], dim=-1)

            logits = postprocess_next_token_scores(
                tokenizer=tokenizer,
                scores=logits,
                input_ids=prev_output_tokens,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                bad_words_ids=[[0]],
                cur_len=gen_len,
                min_length=args.min_generation_length,
                max_length=args.max_generation_length,
                eos_token_id=tokenizer.sep_id,
                repetition_penalty=args.repetition_penalty,
                batch_size=batch_size,
                num_beams=1,
                antonym_dict=antonym_dict
            )

            logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
            # next_token = torch.argmax(logits, dim=-1)
            probs = F.softmax(logits.float(), dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)

            tokens_to_add = next_token * unfinished_sents + tokenizer.sep_id * (1 - unfinished_sents)
            
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


def generate_beam(model_batch, full_context, model, tokenizer: EVATokenizer, args, device):
    '''
        Since the context in model batch is truncated, we need full_context to store the tokens in the entire context.
    '''
    num_beams = args.num_beams
    target_length = args.max_generation_length
    
    do_sample = args.top_p > 0 or args.top_k > 0
    vocab_size = tokenizer.vocab_size
    
    enc_input_ids = model_batch['enc_input_ids']
    enc_attention_mask = model_batch['enc_attention_mask']
    
    enc_input_length = enc_input_ids.size(-1)
    batch_size = enc_input_ids.size(0)
    enc_input_ids = enc_input_ids.unsqueeze(1).expand(batch_size, num_beams, enc_input_length)
    enc_attention_mask = enc_attention_mask.unsqueeze(1).expand(batch_size, num_beams, 1, enc_input_length, enc_input_length)
    
    enc_input_ids = enc_input_ids.contiguous().view(batch_size * num_beams, enc_input_length)
    enc_attention_mask = enc_attention_mask.contiguous().view(batch_size * num_beams, 1, enc_input_length, enc_input_length)
    
    full_context = full_context.unsqueeze(1).expand(batch_size, num_beams, full_context.size(-1))
    full_context = full_context.contiguous().view(batch_size * num_beams, full_context.size(-1))
    
    enc_outputs = model(
        enc_input_ids=enc_input_ids,
        enc_attention_mask=enc_attention_mask,
        only_encoder=True
    )
    enc_hidden_states = enc_outputs["encoder_last_hidden_state"]

    dec_init_length = 1 # 1 for s_0
    # for generating responses
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
        dec_outputs = model(
            dec_input_ids=dec_input_ids,
            dec_attention_mask=dec_attention_mask,
            cross_attention_mask=cross_attention_mask,
            enc_hidden_states=enc_hidden_states,
            past_key_values=past_key_values,
        )
        past_key_values = dec_outputs['past_key_values']
        lm_logits = dec_outputs["lm_logits"]

        gathered_lm_logits = [torch.zeros_like(lm_logits).to(device) for _ in range(mpu.get_model_parallel_world_size())]
        torch.distributed.all_gather(gathered_lm_logits, lm_logits.data, mpu.get_model_parallel_group())
        lm_logits = torch.cat(gathered_lm_logits, dim=-1)

        logits = lm_logits[:, -1, :] / args.temperature
        scores = F.log_softmax(logits, dim=-1)

        prev_output_tokens = torch.cat([full_context, output_ids], dim=-1)

        scores = postprocess_next_token_scores(
            tokenizer=tokenizer,
            scores=scores,
            input_ids=prev_output_tokens,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            bad_words_ids=None,
            cur_len=gen_len,
            min_length=args.min_generation_length,
            max_length=args.max_generation_length,
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
            done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(
                next_scores[batch_idx].max().item(), gen_len
            )

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

        # past_key_values = num_layer * 2 * (2, beam_size, 32, prefix_len, 64) first 2: self/cross attention, second 2: key/value
        past_key_values = [[torch.index_select(layer_past_type, 1, beam_idx) for layer_past_type in layer_past] for layer_past in past_key_values]
        
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
        best_hyp = sorted_hyps.pop()[1]
        best.append(tokenizer.decode(best_hyp.cpu().tolist()))
        best_ids.append(best_hyp.cpu().tolist())

    return best, best_ids
