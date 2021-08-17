# coding=utf-8
# Copyright 2018 The Open AI Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for OpenAI GPT."""
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import json
from io import open
import jieba
import collections
import six


try:
    from functools import lru_cache
except ImportError:
    # Just a dummy decorator to get the checks to run on python2
    # because honestly I don't want to support a byte-level unicode BPE tokenizer on python 2 right now.
    def lru_cache():
        return lambda func: func


def convert_to_unicode(text):
  """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
  if six.PY3:
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  elif six.PY2:
    if isinstance(text, str):
      return text.decode("utf-8", "ignore")
    elif isinstance(text, unicode):
      return text
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))
  else:
    raise ValueError("Not running on Python2 or Python 3?")


def load_vocab(vocab_file):
  """Loads a vocabulary file into a dictionary."""
  vocab = collections.OrderedDict()
  index = 0
  with open(vocab_file, "r") as reader:
    while True:
      token = convert_to_unicode(reader.readline())
      if not token:
        break
      token = token.strip()
      vocab[token] = index
      index += 1
  return vocab


class WordpieceTokenizer(object):

    def __init__(self, vocab, unk_token="<unk>", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, token):

        token = convert_to_unicode(token)

        chars = list(token)
        if len(chars) > self.max_input_chars_per_word:
            return [self.unk_token]

        start = 0
        sub_tokens = []
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if substr in self.vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                sub_tokens.append(self.unk_token)
                start += 1
                continue
            sub_tokens.append(cur_substr)
            start = end

        return sub_tokens


class EncDecTokenizer(object):

    def __init__(self, vocab_file, max_len=None, max_sentinels=187):
        self.max_len = max_len if max_len is not None else int(1e12)
        self.encoder = load_vocab(vocab_file)
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.encoder)

        self.translator = str.maketrans(" \n", "\u2582\u2583")
        self.punct_translator = str.maketrans("！？＂〝〞“”‟＃＄％＆＇‘’‛（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～", "!?\"\"\"\"\"\"#$%&''''()*+,-/:;<=>@[\]^_`{|}~")
        #self.punct_translator = str.maketrans("！？＂", '''!?"''')

        self.sentinel_list = [self.encoder['<s_{}>'.format(i)] for i in range(max_sentinels)]

    @property
    def vocab_size(self):
        return len(self.encoder)

    def __len__(self):
        return len(self.encoder)

    @property
    def sep_id(self):
        return self.encoder[self.sep_token]

    @property
    def pad_id(self):
        return self.encoder[self.pad_token]

    @property
    def cls_id(self):
        return self.encoder[self.cls_token]

    @property
    def go_id(self):
        return self.encoder[self.go_token]

    @property
    def sep_token(self):
        return '<sep>'

    @property
    def pad_token(self):
        return '<pad>'

    @property
    def cls_token(self):
        return '<cls>'

    @property
    def go_token(self):
        return '<go>'

    @property
    def eod_id(self):
        return self.encoder[self.eod_token]

    @property
    def eod_token(self):
        return '<eod>'

    def get_sentinel_num(self):
        return len(self.sentinel_list)

    def get_sentinel_id(self, idx):
        return self.sentinel_list[idx]

    def tokenize(self, text):
        """ Tokenize a string. """
        text = text.replace('…', '...')
        text = text.translate(self.punct_translator)
        output_tokens = []
        for x in jieba.cut(text, cut_all=False):
            x = x.translate(self.translator)
            output_tokens.extend(self.wordpiece_tokenizer.tokenize(x))
        return output_tokens

    def encode(self, text):
        res = [self.encoder[x] for x in self.tokenize(text)]
        return res

    def decode(self, tokens):
        text = ''.join([self.decoder[x] for x in tokens])
        text = text.replace('\u2582', ' ').replace('\u2583', '\n')
        return text
