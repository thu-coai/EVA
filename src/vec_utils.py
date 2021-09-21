# We reuse a fraction of code in http://bitbucket.org/omerlevy/hyperwords.
# Using the numpy and similarity matrix largely speed up the evaluation process,
# compared with evaluation scripts in word2vec and GloVe
# https://github.com/Embedding/Chinese-Word-Vectors

import numpy as np
import argparse
from fuzzywuzzy import fuzz
from tokenization_enc_dec import EncDecTokenizer
import random
from sklearn.metrics.pairwise import cosine_similarity
import os

def read_vectors(path, topn):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = {}
    iw = []
    wi = {}
    with open(path, encoding='utf-8', errors='ignore') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            iw.append(tokens[0])
            if topn != 0 and lines_num >= topn:
                break
    for i, w in enumerate(iw):
        wi[w] = i
    return vectors, iw, wi, dim

def normalize(matrix):
    norm = np.sqrt(np.sum(matrix * matrix, axis=1))
    matrix = matrix / norm[:, np.newaxis]
    return matrix

topn = 0
vectors_path = "/dataset/f1d6ea5b/yjz/sgns.sogounews.bigram-char"
vectors, iw, wi, dim = read_vectors(vectors_path, topn)  # Read top n word vectors. Read all vectors when topn is 0

# Turn vectors into numpy format and normalize them
matrix = np.zeros(shape=(len(iw), dim), dtype=np.float32)
for i, word in enumerate(iw):
	matrix[i, :] = vectors[word]
matrix = normalize(matrix)

def cal_vec_match(str1, str2, tokenizer: EncDecTokenizer):
	l1 = tokenizer.encode(str1)
	l2 = tokenizer.encode(str2)
	vec_avg1 = np.zeros(shape=(dim,), dtype = np.float32)
	vec_avg2 = np.zeros(shape=(dim,), dtype = np.float32)
	for i in l1:
		if(tokenizer.decode([i]) in wi.keys()):
			vec_avg1 += matrix[wi[tokenizer.decode([i])]]
		else:
			return cal_fuzz_match(str1, str2)
	for i in l2:
		if(tokenizer.decode([i]) in wi.keys()):
			vec_avg2 += matrix[wi[tokenizer.decode([i])]]
		else:
			return cal_fuzz_match(str1, str2)
	vec_avg1 /= (len(l1) + 0.00001)
	vec_avg2 /= (len(l2) + 0.00001)
	score = cosine_similarity([vec_avg1, vec_avg2])[0][1]
	# print(str1, " ", str2, "向量相似度：", score)
	return score

def cal_fuzz_match(str1, str2):
    # print("cal_fuzz_match" , str1, str2)
    return fuzz.token_sort_ratio(str1, str2) / 100