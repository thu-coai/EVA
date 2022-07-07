import torch
from arguments import get_args
from model import EVAConfig, EVAModel, EVATokenizer
from utils import set_random_seed

args = get_args()
device = torch.cuda.current_device()

set_random_seed(args.seed)

tokenizer = EVATokenizer.from_pretrained("/home/guyuxian/EVA/checkpoints/eva2.0-hf")
model = EVAModel.from_pretrained("/home/guyuxian/EVA/checkpoints/eva2.0-hf")
model = model.to(device)

input_str = "今天天气怎么样？"

input_ids = tokenizer(input_str, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids.to(device)

print(input_ids)
gen = model.generate(input_ids, do_sample=True, decoder_start_token_id=tokenizer.bos_token_id, top_p=0.9, max_length=32, use_cache=True)

print(tokenizer.decode(gen[0].cpu().tolist()))

print(gen)

# input_tokens = tokenizer.encode(input_str) + [tokenizer.sep_id, tokenizer.get_sentinel_id(0)]
# input_tokens = torch.tensor(input_tokens).unsqueeze(0).to(device)

# model_batch = get_inference_batch(input_tokens, device, 1, tokenizer, args)

# print(input_tokens)
# print(model_batch)
# with torch.no_grad():
#     generation_str_list, generation_id_list = generate_no_beam(model_batch, input_tokens, model, tokenizer, args, device)

# print(generation_str_list)