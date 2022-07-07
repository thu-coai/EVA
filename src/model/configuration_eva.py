# coding=utf-8

""" Encoder-Decoder model configuration """

import json
import os
import copy
from transformers import PretrainedConfig
from typing import Any, Dict, Union

class EVAConfig(PretrainedConfig):
    model_type = "eva"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}
    def __init__(
        self,
        d_model=768,
        d_kv=64,
        d_ff=256,
        num_layers=12,
        num_decoder_layers=12,
        num_heads=12,
        relative_attention_num_buckets=32,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="relu",
        use_cache=False,
        use_scaled_init_for_output_weights=True,
        init_method_std=0.02,
        max_position_embeddings=1024,
        attn_scale=False,
        vocab_size=30000,
        is_encoder_decoder=True,
        pad_token_id=5,
        eos_token_id=4,
        **kwargs
    ):
        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            **kwargs
        )
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
        self.use_scaled_init_for_output_weights = use_scaled_init_for_output_weights
        self.init_method_std = init_method_std
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size
        self.attn_scale = attn_scale
        self.is_encoder_decoder = is_encoder_decoder
