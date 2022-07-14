from model import EVAModel, EVATokenizer

tokenizer = EVATokenizer.from_pretrained("/PATH-TO-EVA-CHECKPOINT/")
model = EVAModel.from_pretrained("/PATH-TO-EVA-CHECKPOINT/")
model = model.half().cuda()

input_str = "今天天气怎么样"

tokenize_out = tokenizer(input_str, "", return_tensors="pt", padding=True, truncation=True, max_length=512)
input_ids = tokenize_out.input_ids.cuda()

gen = model.generate(input_ids, do_sample=True, decoder_start_token_id=tokenizer.bos_token_id, top_p=0.9, max_length=32, use_cache=True)
print(tokenizer.decode(gen[0], skip_special_tokens=True))
