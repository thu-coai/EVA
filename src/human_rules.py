from tokenization_enc_dec import EncDecTokenizer
import random
import re
import os
from fuzzywuzzy import fuzz
from text_match import cal_match, new_cal_match
from vec_utils import cal_vec_match
import copy

greetings = [
(
    ["你好", "您好", "Hello", "hello"], 
    ["我是一个小机器人 EVA。",
    "你好，我的名字叫 EVA。",
    "您好，我的名字叫 EVA，不是那个 EVA 哦！",
    "您好，我是 EVA，今天天气不错，想不想一起出去玩儿",
    "您好，很高兴见到您，EVA 向您问好"]
),
]

in_context_rules = []
repeition_resp = []
continue_resp = []

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
low_quality_min_length = 20
low_quality_re_list = [
    r'.?哈哈哈.?',
    r'.?!!!.?!!!.?',
    r'.?啊啊啊.?!!!',
    r'.?啊啊啊.?哈哈哈.?',
    r'.?。。。.?。。。.?',
    r'.?<.?>.?',
    r'好可爱',
    r'笑死我',
    r'好棒',
    r'那就好',
    r'。。。',
]


def init_list():
    # in_context_rules
    all_rules = []
    for path, dir_list, file_list in os.walk("/dataset/f1d6ea5b/gyx-eva/eva-origin/rules/in_context"):
        for file_name in file_list:
            with open(os.path.join(path, file_name)) as f:
                rules = f.readlines()
                all_rules.extend(rules)
    # l = open("/dataset/f1d6ea5b/yjz/eva-origin/src/chatterbot.tsv", "r").read().split("\n")
    all_rules = [x.strip() for x in all_rules]
    for rule in all_rules:
        # if len(rule.split("\t")) != 2:
        #     print(rule)
        post, resp = rule.split("\t")
        posts = post.split("|")
        resps = resp.split("|")
        in_context_rules.append((posts, resps))

    # repeition_resp
    # 0: normal
    # 1: question
    # 2: bye-bye
    with open("/dataset/f1d6ea5b/gyx-eva/eva-origin/rules/repetition/resp.txt") as f:
        lines = f.readlines()
    for line in lines:
        repeition_resp.append(line.strip().split("|"))

    # continue_resp
    with open("/dataset/f1d6ea5b/gyx-eva/eva-origin/rules/in_context/continue.txt") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        continue_resp.extend(line[1].split("|"))

    # l1, l2 = [], []
    # in_l1, in_l2 = [], []
    # for i in l:
    #     if(len(i.split("\t")) == 2):
    #         l1.append(i.split("\t")[0])
    #         l2.append(i.split("\t")[1])
    # last_input = ""
    # for i in range(len(l1)):
    #     if l1[i] == last_input:
    #         in_l2.append(l2[i])
    #     else:
    #         if(in_l1 != []):
    #             in_context.append((in_l1, in_l2))
    #         in_l1, in_l2 = [], []
    #         in_l1.append(l1[i])
    #         in_l2.append(l2[i])
    #         last_input = l1[i]


def check_resp(output_tokens, tokenizer: EncDecTokenizer):
    """
    需要重新生成时，返回true
    """
    # return False
    output_texts = tokenizer.decode(output_tokens)
    # print("output_texts = ", output_texts, "len = ", len(output_texts))
    for priv_re in priv_re_list:
        if(re.search(priv_re, output_texts) != None):
            # print("检查到涉及个人隐私的词汇")
            return True
    if(len(output_texts)<low_quality_min_length):
        for lq_re in low_quality_re_list:
            # print(lq_re)
            if(re.search(lq_re, output_texts) != None):
                # print("检查到低质量生成")
                return True
    return False


def get_resp(all_contexts_str, input_text, tokenizer: EncDecTokenizer):
    usr_contexts_str = [x for i, x in enumerate(all_contexts_str) if i % 2 == 0]
    sys_contexts_str = [x for i, x in enumerate(all_contexts_str) if i % 2 == 1]

    # begining of he dialog
    if len(all_contexts_str) == 1:
        for g in greetings:
            for p in g[0]:
                if cal_match(p, input_text) > 0.8:
                    resp = random.choice(g[1])
                    return {"resp": resp, "continue": False}
    
    # usr repetition
    num = sum(1 if fuzz.token_sort_ratio(x, input_text) > 60 else 0 for x in usr_contexts_str[-5:])
    if num > 3:
        best_resp = handle_usr_repetition(input_text, num, sys_contexts_str)
    else:
        waiting_list = []
        for g in in_context_rules:
            for p in g[0]:
                # if fuzz.token_sort_ratio(p, input_text) > 60:
                vec_match_score = cal_vec_match(p, input_text, tokenizer)
                # print("vec_match_score", vec_match_score)
                if vec_match_score > 0.8:
                    waiting_list.append(g)
                    break
        best_resp = find_best(waiting_list, input_text, usr_contexts_str, sys_contexts_str)
    
    if best_resp is None:
        return None
    else:
        # print(continue_resp)
        if best_resp in continue_resp:
            return {"resp": best_resp, "continue": True}
        else:
            return {"resp": best_resp, "continue": False}


def is_question(input_text):
    a = (input_text[-1] in ["?", "？"])
    b = (input_text[-1] in ["吗"])

    return a or b


def handle_sys_repetition(input_text, cands, sys_contexts_str):
    cands_copy = copy.deepcopy(cands)
    random.shuffle(cands_copy)
    for r in cands_copy:
        if r not in sys_contexts_str:
            return r
    else:
        return None


def handle_usr_repetition(input_text, num, sys_contexts_str):
    if num > 4:
        return handle_sys_repetition(input_text, repeition_resp[2], sys_contexts_str)
    if is_question(input_text):
        return handle_sys_repetition(input_text, repeition_resp[1], sys_contexts_str)
    
    return handle_sys_repetition(input_text, repeition_resp[0], sys_contexts_str)


def find_best(waiting_list, input_text, usr_contexts_str, sys_contexts_str):
    if len(waiting_list) == 0:
        return None
    best_score = 0
    best_resp = None
    for g in waiting_list:
        for p in g[0]:
            temp_score = cal_match(p, input_text)
            # print(input_text, " ", p, " ", temp_score)
            if temp_score > best_score:
                best_resp = g
                best_score = temp_score

    if best_resp == None or best_score < 0.5:
        return None
    # return find_final_resp(input_text, best_resp)
    return handle_sys_repetition(input_text, best_resp[1], sys_contexts_str)


# def find_final_resp(input_text, best_resp):
#     res_list = []
#     best_score = 0
#     for i in best_resp[1]:
#         temp_score = cal_match(input_text, i)
#         if(temp_score > best_score):
#             best_score = temp_score
#             res_list = []
#             res_list.append(i)
#         elif(temp_score == best_score):
#             res_list.append(i)
#     print(res_list)
#     return random.choice(res_list)

        
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
