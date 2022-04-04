# coding=utf-8

"""Inference EVA"""

from lib2to3.pgen2 import token
import os
import torch
import torch.nn.functional as F
from arguments import get_args
from model import EVAConfig,EVATokenizer,EVAModel
import deepspeed
import torch.distributed as dist

from utils import set_random_seed
from generation_utils import generate_beam, generate_no_beam
    


def get_attn_masks(args,
                  tokenizer,
                  contexts,
                  targets):
    # Extract batch size and sequence length.
    batch_size, enc_seq_length = contexts.size()

    # Enc Attention mask.
    enc_attn_mask = torch.zeros(
        batch_size, 1, enc_seq_length, enc_seq_length, device=contexts.device)

    ctx_lengths = (contexts != tokenizer.pad_id).sum(1)
    for b in range(batch_size):
        enc_attn_mask[b, 0, :ctx_lengths[b], :ctx_lengths[b]] = 1

    batch_size, dec_seq_length = targets.size()
    # Dec Attention mask
    dec_attn_mask = torch.tril(torch.ones(
        batch_size, 1, dec_seq_length, dec_seq_length, device=targets.device))

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
        "dec_attention_mask": dec_attn_mask,
        "cross_attention_mask": cross_attn_mask,
    }

    return model_batch


def get_inference_batch(
        context_tokens,
        device,
        batch_size,
        tokenizer,
        args,
    ):
    tokens = context_tokens
    tokens = tokens.view(batch_size, -1).contiguous()
    tokens = tokens.to(device)
    
    targets = torch.zeros(batch_size, 1, dtype=torch.long, device=device) + tokenizer.get_sentinel_id(0)

    # Get the masks and postition ids.
    model_batch = get_attn_masks(
        args,
        tokenizer,
        tokens,
        targets,
    )
    
    model_batch = {
        "enc_input_ids": tokens,
        "dec_input_ids": targets,
        **model_batch
    }

    return model_batch


def generate_samples(model, tokenizer: EVATokenizer, args, device):
    model.eval()

    with torch.no_grad():
        full_context_list = []
        while True:
            input_text = input("Usr >>> ")
            if input_text == "clear":
                print("Clear Dialog")
                # set_random_seed(args.seed) # reset rng
                full_context_list = []
                continue
            if input_text == "seed":
                seed = int(input("Seed >>> "))
                print("Clear Dialog")
                set_random_seed(seed)
                full_context_list = []
                continue
            else:
                full_context_list.append(tokenizer.encode(input_text) + [tokenizer.sep_id])
                full_context = [x for y in full_context_list for x in y]
                trunc_context = []
                for utt in full_context_list[:-9:-1]:
                    if len(trunc_context) + len(utt) + 1 <= 128:
                        trunc_context = utt + trunc_context
                trunc_context.append(tokenizer.get_sentinel_id(0))
                trunc_context = torch.tensor(trunc_context, dtype=torch.long).to(device)
                full_context = torch.tensor(full_context, dtype=torch.long).to(device)
            
            # encoder tensor
            trunc_context = trunc_context.unsqueeze(0).repeat(args.batch_size, 1) # repeat
            full_context = full_context.unsqueeze(0).repeat(args.batch_size, 1) 
            model_batch = get_inference_batch(trunc_context, device, args.batch_size, tokenizer, args)
   
            if args.num_beams == 1:
                generation_str_list, generation_id_list = generate_no_beam(model_batch, full_context, model, tokenizer, args, device)
            else:
                generation_str_list, generation_id_list  = generate_beam(model_batch, full_context, model, tokenizer, args, device)

            full_context_list.append(generation_id_list[0] + [tokenizer.sep_id])

            print("Sys >>> {}".format(generation_str_list[0]))


def main():
    """Main serving program."""

    print('Loading Model ...')
    args = get_args()
    set_random_seed(args.seed)
    # Disable CuDNN.
    tokenizer = EVATokenizer(os.path.join(args.tokenizer_path, 'vocab.txt'))
    config = EVAConfig.from_json_file(args.model_config)
    config.vocab_size = tokenizer.vocab_size
    config.feed_forward_proj = "gated-gelu"
    model = EVAModel.from_pretrained(args.load,config=config)
    args.batch_size = 1
    print(args)
    # os.system('clear')
    print('Model Loaded!')
    #generate samples
    generate_samples(model, tokenizer, args, torch.cuda.current_device())
    

if __name__ == "__main__":
    main()



