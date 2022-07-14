import torch
from arguments import get_args
from model import EVAModel, EVATokenizer
from utils import set_random_seed

args = get_args()
device = torch.cuda.current_device()

set_random_seed(args.seed)

tokenizer = EVATokenizer.from_pretrained("/home/guyuxian/EVA/checkpoints/eva2.0-hf")
model = EVAModel.from_pretrained("/home/guyuxian/EVA/checkpoints/eva2.0-hf")
model = model.to(device)
model = model.half()

input_str = ["今天天气怎么样？<sep>我也不知道哎，你知道吗？", "你今天吃的啥？<sep>我吃了两个汉堡"]
# labels = ["这几天天气都不错。不过很干燥", "吃了两个小时面包加两根香肠"]

tokenize_out = tokenizer(input_str, ["" for _ in input_str], return_tensors="pt", padding=True, truncation=True, max_length=512)
input_ids = tokenize_out.input_ids.to(device)
attention_mask = tokenize_out.attention_mask.to(device)

print(tokenize_out)

# label_tokenize_out = tokenizer(labels, return_tensors="pt", padding=True, truncation=True, max_length=512)
# labels = label_tokenize_out.input_ids.to(device)
# decoder_attention_mask = label_tokenize_out.attention_mask.to(device)
# labels = torch.where(labels==tokenizer.pad_token_id, -100, labels)

# print(attention_mask)
# print(decoder_attention_mask)

# output = model(input_ids, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask, labels=labels)

# print(output.loss)

gen = model.generate(input_ids, attention_mask=attention_mask, do_sample=True, decoder_start_token_id=tokenizer.bos_token_id, top_p=0.9, max_length=32, use_cache=True)

print(gen)
print(tokenizer.batch_decode(gen, skip_special_tokens=True))
