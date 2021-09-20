import torch
from transformers import XLNetTokenizer
from hanziconv import HanziConv
import torch
from torch import nn
from transformers import XLNetForSequenceClassification, XLNetConfig

class XlnetModel(nn.Module):
    def __init__(self):
        super(XlnetModel, self).__init__()
        self.xlnet = XLNetForSequenceClassification.from_pretrained("hfl/chinese-xlnet-base", num_labels = 2)  # /bert_pretrain/
        self.device = torch.device("cuda")
        for param in self.xlnet.parameters():
            param.requires_grad = True  # 每个参数都要 求梯度

    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.xlnet(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities
    
    
class XlnetModelTest(nn.Module):
    def __init__(self):
        super(XlnetModelTest, self).__init__()
        config = XLNetConfig.from_pretrained('/dataset/f1d6ea5b/yjz/eva-origin/src/models/config.json')
        self.xlnet = XLNetForSequenceClassification(config)  # /bert_pretrain/
        self.device = torch.device("cuda:0")
    def forward(self, batch_seqs, batch_seq_masks, batch_seq_segments, labels):
        loss, logits = self.xlnet(input_ids = batch_seqs, attention_mask = batch_seq_masks, 
                              token_type_ids=batch_seq_segments, labels = labels)[:2]
        probabilities = nn.functional.softmax(logits, dim=-1)
        return loss, logits, probabilities

max_seq_len = 400

def trunate_and_pad(tokens_seq_1, tokens_seq_2):
	"""
	1. 如果是单句序列，按照BERT中的序列处理方式，需要在输入序列头尾分别拼接特殊字符'CLS'与'SEP'，
		因此不包含两个特殊字符的序列长度应该小于等于max_seq_len-2，如果序列长度大于该值需要那么进行截断。
	2. 对输入的序列 最终形成['CLS',seq,'SEP']的序列，该序列的长度如果小于max_seq_len，那么使用0进行填充。
	入参: 
		seq_1       : 输入序列，在本处其为单个句子。
		seq_2       : 输入序列，在本处其为单个句子。
		max_seq_len : 拼接'CLS'与'SEP'这两个特殊字符后的序列长度
	
	出参:
		seq         : 在入参seq的头尾分别拼接了'CLS'与'SEP'符号，如果长度仍小于max_seq_len，则使用0在尾部进行了填充。
		seq_mask    : 只包含0、1且长度等于seq的序列，用于表征seq中的符号是否是有意义的，如果seq序列对应位上为填充符号，
						那么取值为1，否则为0。
		seq_segment : shape等于seq，单句，取值都为0 ，双句按照01切分
		
	"""
	# 对超长序列进行截断
	if len(tokens_seq_1) > ((max_seq_len - 3)//2):
		tokens_seq_1 = tokens_seq_1[0:(max_seq_len - 3)//2]
	if len(tokens_seq_2) > ((max_seq_len - 3)//2):
		tokens_seq_2 = tokens_seq_2[0:(max_seq_len - 3)//2]
	# 分别在首尾拼接特殊符号
	seq = tokens_seq_1 + ['<sep>'] + tokens_seq_2 + ['<sep>'] + ['<cls>']
	seq_segment = [0] * (len(tokens_seq_1) + 1) + [1] * (len(tokens_seq_2) + 1) + [2]
	# ID化
	seq = bert_tokenizer.convert_tokens_to_ids(seq)
	# 根据max_seq_len与seq的长度产生填充序列
	padding = [0] * (max_seq_len - len(seq))
	# 创建seq_mask
	seq_mask = [1] * len(seq) + padding
	# 创建seq_segment
	seq_segment = seq_segment + padding
	# 对seq拼接填充序列
	seq += padding
	assert len(seq) == max_seq_len
	assert len(seq_mask) == max_seq_len
	assert len(seq_segment) == max_seq_len
	return seq, seq_mask, seq_segment

device = torch.device("cuda:0")
torch.cuda.set_device(torch.device("cuda", 0))
bert_tokenizer = XLNetTokenizer.from_pretrained('/dataset/f1d6ea5b/yjz/eva-origin/src/models/spiece.model', do_lower_case=True)
checkpoint = torch.load("/dataset/f1d6ea5b/yjz/eva-origin/src/models/best.pth.tar")
model = XlnetModelTest().to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

def cal_match(str1, str2):
	s1 = map(HanziConv.toSimplified, [str1])
	s2 = map(HanziConv.toSimplified, [str2])
	tokens_seq_1 = list(map(bert_tokenizer.tokenize, s1))
	tokens_seq_2 = list(map(bert_tokenizer.tokenize, s2))
	result = list(map(trunate_and_pad, tokens_seq_1, tokens_seq_2))
	seqs = [i[0] for i in result]
	seq_masks = [i[1] for i in result]
	seq_segments = [i[2] for i in result]
	seqs = torch.Tensor(seqs).type(torch.long).to(device)
	masks = torch.Tensor(seq_masks).type(torch.long).to(device)
	segments = torch.Tensor(seq_segments).type(torch.long).to(device)
	labels = torch.Tensor([0.8]).type(torch.long).to(device)
	_, _, p = model(seqs, masks, segments, labels)
	score =  p.to("cpu")[0][1].item()
	# print(str1, " ", str2, " ",score)
	return score


def new_cal_match(str1, in_context):
	s1 = map(HanziConv.toSimplified, [str1 for i in range(len(in_context))])
	s2 = map(HanziConv.toSimplified, [in_context[i][0][0] for i in range(len(in_context))])
	tokens_seq_1 = list(map(bert_tokenizer.tokenize, s1))
	tokens_seq_2 = list(map(bert_tokenizer.tokenize, s2))
	result = list(map(trunate_and_pad, tokens_seq_1, tokens_seq_2))
	seqs = [i[0] for i in result]
	seq_masks = [i[1] for i in result]
	seq_segments = [i[2] for i in result]
	seqs = torch.Tensor(seqs).type(torch.long).to(device)
	masks = torch.Tensor(seq_masks).type(torch.long).to(device)
	segments = torch.Tensor(seq_segments).type(torch.long).to(device)
	labels = torch.Tensor([0.8 for i in range(len(in_context))]).type(torch.long).to(device)
	with torch.no_grad():
		_, _, p = model(seqs, masks, segments, labels)
	new_p = p[:, 1]
	index = torch.argmax(new_p, dim=0)
	print("概率为", new_p[index])
	return index, new_p[index]

# test code

# while(True):
# 	str1 = input("str1 = ")
# 	str2 = input("str2 = ")
# 	print(str1, " ", str2, " ", cal_match(str1, str2))

