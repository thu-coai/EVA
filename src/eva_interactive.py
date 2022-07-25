# coding=utf-8

"""Inference EVA"""

import torch
from arguments import get_args
from model import EVATokenizer, EVAModel

from utils import set_random_seed


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
                full_context_list.append(tokenizer.encode(input_text))
                trunc_context = []
                for utt in full_context_list[:-9:-1]:
                    if len(trunc_context) + len(utt) + 1 <= 128:
                        trunc_context = utt + trunc_context
                trunc_context.append(tokenizer.bos_token_id)
                trunc_context = torch.tensor(trunc_context, dtype=torch.long).unsqueeze(0).to(device)
            
            model_gen = model.generate(
                trunc_context,
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

            model_gen_list = model_gen[0].cpu().tolist()
            model_gen_str = tokenizer.decode(model_gen[0], skip_special_tokens=True)

            full_context_list.append(model_gen_list[1:])

            print("Sys >>> {}".format(model_gen_str))


def main():
    """Main serving program."""

    print('Loading Model ...')
    args = get_args()
    set_random_seed(args.seed)

    device = torch.cuda.current_device()

    tokenizer = EVATokenizer.from_pretrained(args.load)
    model = EVAModel.from_pretrained(args.load)
    model = model.to(device).half()

    # os.system('clear')
    print('Model Loaded!')
    #generate samples
    generate_samples(model, tokenizer, args, device)
    

if __name__ == "__main__":
    main()
