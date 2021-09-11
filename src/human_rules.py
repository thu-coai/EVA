from tokenization_enc_dec import EncDecTokenizer
import random
import re
from fuzzywuzzy import fuzz

greetings = [
(
    ["你好", "您好", "Hello", "hello"], 
    ["我是一个小机器人 EVA。",
    "你好，我的名字叫 EVA。",
    "您好，我的名字叫 EVA，不是那个 EVA 哦！",
    "您好，我是 EVA，今天天气不错，想不想一起出去玩儿",
    "您好，很高兴见到您，EVA 向您问好"]
),
(
    ["今天天气真好", "今天天气真不错"],
    [
        "是呀是呀，风也不大，太阳高高挂嘻嘻",
        "趁着好天气一起出去玩儿吧！",
        "很久没有这么好的天气了",
        "北京的秋天是最美丽的"]
),
]

in_context = [
(
    ["天气怎么样", "天气如何", "天气咋样", "天气什么样"],
    ["很好啊，是北京难得的好天气", "很好呀，阳光灿烂，非常适合出去玩儿", "哈哈哈我可不是天气预报小姐哦，但是感觉应该是不错呢"]
)
]

simple_replace = [
    (
        ["好吧", "好吧。", "好吧。。。", "哈哈哈,好吧", "哈哈哈，好吧。"],
        ["好吧。您还有啥想知道的吗？", "好吧，我得吃饭去了，改天再聊~", "好吧，祝您度过美好的一天！"]
    )
]

# 个人隐私regex
priv_re_list = [
    r'^\w+([-+.]\w+)*@\w+([-.]\w+)*\.\w+([-.]\w+)*$', # 邮箱
    r'^(13[0-9]|14[5|7]|15[0|1|2|3|5|6|7|8|9]|18[0|1|2|3|5|6|7|8|9])\d{8}$' # 手机号码
    r'^((\d{3,4}-)|\d{3.4}-)?\d{7,8}$', # 电话号码("XXX-XXXXXXX"、"XXXX-XXXXXXXX"、"XXX-XXXXXXX"、"XXX-XXXXXXXX"、"XXXXXXX"和"XXXXXXXX)
    r'\d{3}-\d{8}|\d{4}-\d{7}', # 同电话号码
    r'^\d{15}|\d{18}$', # 身份证号
    r'^([0-9]){7,18}(x|X)?$', # 以X结尾身份证号
    r'[1-9][0-9]{4,}', # qq号
    r'[1-9]\d{5}(?!\d)', # 邮政编码
    r'\d+\.\d+\.\d+\.\d+', # ip地址
    r'((?:(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d)\\.){3}(?:25[0-5]|2[0-4]\\d|[01]?\\d?\\d))', # ip地址
]

# 需要同时满足以下两个条件：长度小于low_quality_min_length且被正则匹配，进行重新生成
low_quality_min_length = 7
low_quality_re_list = [
    r'.+哈哈哈.+',
    r'.+!!!.+!!!.+',
    r'.+啊啊啊.+!!!',
    r'.+啊啊啊.+哈哈哈.+',
    r'.+。。。.+。。。.+',
    r'.+<.+>.+',
    r'好可爱',
    r'笑死我',
    r'好棒',
    r'那就好',
    r'。。。',
]


def init_list():
    """
    初始化in_context
    """
    l = open("/dataset/f1d6ea5b/yjz/eva-origin/src/chatterbot.tsv", "r").read().split("\n")
    l1, l2 = [], []
    in_l1, in_l2 = [], []
    for i in l:
        if(len(i.split("\t")) == 2):
            l1.append(i.split("\t")[0])
            l2.append(i.split("\t")[1])
    last_input = ""
    for i in range(len(l1)):
        if l1[i] == last_input:
            in_l2.append(l2[i])
        else:
            in_context.append((in_l1, in_l2))
            in_l1, in_l2 = [], []
            in_l1.append(l1[i])
            in_l2.append(l2[i])
            last_input = l1[i]

def check_resp(output_tokens, tokenizer: EncDecTokenizer):
    """
    需要重新生成时，返回true
    """
    output_texts = tokenizer.decode(output_tokens)
    for priv_re in priv_re_list:
        if(re.search(priv_re, output_texts) != None):
            # print("检查到涉及个人隐私的词汇")
            return True
    if(len(output_texts)<low_quality_min_length):
        for lq_re in low_quality_re_list:
            if(re.search(lq_re, output_texts) != None):
                # print("检查到低质量生成")
                return True
    return False

def get_resp(all_input_tokens, input_text, tokenizer: EncDecTokenizer):
    all_input_texts = tokenizer.decode(all_input_tokens)
    contexts = all_input_texts.split("<sep>")

    if len(all_input_tokens) == 0:
        for g in greetings:
            for p in g[0]:
                if p in input_text:
                    resp = random.choice(g[1])
                    return resp
    waiting_list = []
    for g in in_context:
        for p in g[0]:
            if p in input_text:
                waiting_list.append(g)
    return find_best(waiting_list, input_text)

def find_best(waiting_list, input_text):
    if len(waiting_list) == 0:
        return None
    best_score = 0
    best_resp = None
    for g in waiting_list:
        for p in g[0]:
            temp_score = fuzz.ratio(p, input_text)
            if temp_score > best_score:
                best_resp = g
                best_score = temp_score
    if best_resp == None:
        return None
    return random.choice(best_resp[1]) 
        
def post_process(all_input_tokens, input_text, gen_text_ids, tokenizer: EncDecTokenizer):
    gen_text = tokenizer.decode(gen_text_ids)
    gen_text = gen_text.replace("#e-s[数字x]", "")
    gen_text = gen_text.replace("京东", "CoAI 小组")

    if len(gen_text) < 8:
        for s in simple_replace:
            for p in s[0]:
                if p in gen_text:
                    gen_text = random.choice(s[1])
                    

    gen_text_ids = tokenizer.encode(gen_text)

    return gen_text_ids
    