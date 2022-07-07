# coding=utf-8

import os
import jieba
import collections
import warnings
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import logging
from typing import Any, Dict, List, Optional, Tuple
from shutil import copyfile


logger = logging.get_logger(__name__)


VOCAB_FILES_NAMES = {"vocab_file": "vocab.txt"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "eva": "https://huggingface.co/t5-base/resolve/main/spiece.model",
        "eva2-base": "https://huggingface.co/t5-base/resolve/main/spiece.model",
        "eva2-large": "https://huggingface.co/t5-large/resolve/main/spiece.model",
        "eva2-xlarge": "https://huggingface.co/t5-3b/resolve/main/spiece.model",
    }
}


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
      return text
    elif isinstance(text, bytes):
      return text.decode("utf-8", "ignore")
    else:
      raise ValueError("Unsupported string type: %s" % (type(text)))


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
    
    
class EVATokenizer(PreTrainedTokenizer):
    
    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    model_input_names = ["input_ids", "attention_mask"]
    
    def __init__(
        self,
        vocab_file,
        eos_token="<sep>",
        sep_token="<sep>",
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s_0>",
        max_sentinels=190,
        **kwargs
    ) -> None:
        
        super().__init__(
            eos_token=eos_token,
            unk_token=unk_token,
            pad_token=pad_token,
            sep_token=sep_token,
            bos_token=bos_token,
            max_sentinels=max_sentinels,
            **kwargs
        )
        self.vocab_file = vocab_file
        self.encoder = load_vocab(vocab_file)
        self.decoder = {v:k for k,v in self.encoder.items()}
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.encoder)

        self.translator = str.maketrans(" \n", "\u2582\u2583")
        self.punct_translator = str.maketrans("！？＂〝〞“”‟＃＄％＆＇‘’‛（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～", "!?\"\"\"\"\"\"#$%&''''()*+,-/:;<=>@[\]^_`{|}~")

        self.sentinel_list = [self.encoder['<s_{}>'.format(i)] for i in range(max_sentinels)]
        
    @property
    def vocab_size(self):
        return len(self.encoder)
    
    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        return vocab
    
    def _add_eos_if_not_present(self, token_ids: List[int]) -> List[int]:
        """Do not add eos again if user already added it."""
        if len(token_ids) > 0 and token_ids[-1] == self.eos_token_id:
            warnings.warn(
                f"This sequence already has {self.eos_token}. In future versions this behavior may lead to duplicated"
                " eos tokens being added."
            )
            return token_ids
        else:
            return token_ids + [self.eos_token_id]
    
    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. A sequence has the following format:

        - single sequence: `X </s>`
        - pair of sequences: `A </s> B </s>`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        token_ids_0 = self._add_eos_if_not_present(token_ids_0)
        if token_ids_1 is None:
            return token_ids_0 + [self.get_sentinel_id(0)]
        else:
            token_ids_1 = self._add_eos_if_not_present(token_ids_1)
            return token_ids_0 + token_ids_1 + [self.get_sentinel_id(0)]
    
    def _tokenize(self, text: str) -> List[str]:
        """ Tokenize a string. """
        text = text.replace('…', '...')
        text = text.translate(self.punct_translator)
        output_tokens = []
        for x in jieba.cut(text, cut_all=False):
            x = x.translate(self.translator)
            output_tokens.extend(self.wordpiece_tokenizer.tokenize(x))
        return output_tokens
    
    def _convert_token_to_id(self, token):
        return self.encoder[token]
    
    def _convert_id_to_token(self, index):
        return self.decoder[index]
    
    def convert_tokens_to_string(self, tokens):
        text = ''.join(tokens)
        text = text.replace('\u2582', ' ').replace('\u2583', '\n')
        return text

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(save_directory, (filename_prefix + "-" if filename_prefix else "") + "vocab.txt")

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file) and os.path.isfile(self.vocab_file):
            copyfile(self.vocab_file, out_vocab_file)
        elif not os.path.isfile(self.vocab_file):
            with open(out_vocab_file, "wb") as fi:
                for index in range(self.vocab_size):
                    fi.write(self._convert_id_to_token(index) + "\n")

        return (out_vocab_file,)
    
    def check(self, token):
        return token in self.encoder

    def get_sentinel_num(self):
        return len(self.sentinel_list)

    def get_sentinel_id(self, idx):
        return self.sentinel_list[idx]
