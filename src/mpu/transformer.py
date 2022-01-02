# coding=utf-8

"""Encoder-Decoder Model"""

import math

import torch
import torch.nn as nn
import deepspeed

from .initialize import get_model_parallel_world_size
from .layers import ColumnParallelLinear
from .layers import RowParallelLinear

from .random import checkpoint
from .random import get_cuda_rng_tracker

from .utils import divide
from .utils import split_tensor_along_last_dim

from .layers import VocabParallelEmbedding

from typing import Callable, Optional
from model.configuration_eva import EVAConfig


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states


@torch.jit.script
def gelu_impl(x):
    """OpenAI's gelu implementation."""
    return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                       (1.0 + 0.044715 * x * x)))


def gelu(x):
    return gelu_impl(x)


def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_


def scaled_init_method(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)

    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


def init_method_normal(std):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_


class ParallelDenseReluDense(nn.Module):
    def __init__(self,
                 config: EVAConfig,
                 init_method: Callable,
                 output_layer_init_method: Optional[Callable] = None):
        super(ParallelDenseReluDense, self).__init__()
        self.wi_0 = ColumnParallelLinear(
            config.d_model, config.d_ff,
            gather_output=False,
            bias=False,
            init_method=init_method_normal(config.init_method_std))
        self.wi_1 = ColumnParallelLinear(
            config.d_model, config.d_ff,
            gather_output=False,
            bias=False,
            init_method=init_method_normal(config.init_method_std))
        self.wo = RowParallelLinear(
            config.d_ff,
            config.d_model,
            bias=False,
            input_is_parallel=True,
            init_method=init_method_normal(config.init_method_std))
        self.dropout = nn.Dropout(config.dropout_rate)

        # self.do_dim_trick = config.do_dim_trick
        # if torch.distributed.get_rank() % 5 == 4:
        #     self.ff_mask = nn.Parameter(torch.tensor([1.0] * 13104 + [0.0] * 4), requires_grad=False)
        # else:
        #     self.ff_mask = nn.Parameter(torch.tensor([1.0] * 13108), requires_grad=False)

    def forward(self, hidden_states):
        # hidden_states: [b, s, hp]
        hidden_gelu = gelu(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        # hidden_states: [b, s, d_ff_p]
        # if self.do_dim_trick:
        #     ff_mask = self.ff_mask.view(1, 1, self.ff_mask.size(0))
        #     hidden_states = ff_mask * hidden_states

        # hidden_states = F.relu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        # hidden_states: [b, s, hp]
        return hidden_states


class ParallelAttention(nn.Module):
    def __init__(
        self,
        config: EVAConfig, 
        init_method: Callable,
        is_decoder: bool = False,
        is_cross_attn: bool = False,
        output_layer_init_method: Optional[Callable] = None,
        has_relative_attention_bias: bool = False):
        super(ParallelAttention, self).__init__()

        self.is_decoder = is_decoder
        self.is_cross_attn = is_cross_attn
        self.attn_scale = config.attn_scale

        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets

        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        
        d_attn_out = config.d_kv * config.num_heads # h
        
        # Per attention head and per partition values.
        world_size = get_model_parallel_world_size() # p
        self.hidden_size_per_partition = divide(d_attn_out, world_size) # h_p
        self.hidden_size_per_attention_head = config.d_kv # h_i
        self.num_attention_heads_per_partition = divide(config.num_heads, world_size) # n_p

        # Strided linear layer.
        if is_cross_attn:
            self.project_q = ColumnParallelLinear(config.d_model, d_attn_out,
                                                  stride=1, # NOTE: modify stride
                                                  bias=False,
                                                  gather_output=False,
                                                  init_method=init_method_normal(config.init_method_std))
            self.project_kv = ColumnParallelLinear(config.d_model, 2 * d_attn_out,
                                                   stride=2,  # NOTE: modify stride
                                                   bias=False,
                                                   gather_output=False,
                                                   init_method=init_method_normal(config.init_method_std))
        else:
            self.project = ColumnParallelLinear(config.d_model, 3 * d_attn_out,
                                                        stride=3,
                                                        bias=False,
                                                        gather_output=False,
                                                init_method=init_method_normal(config.init_method_std))
        
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.num_attention_heads_per_partition)
        
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = nn.Dropout(config.dropout_rate)

        # Output.
        self.dense = RowParallelLinear(d_attn_out,
                                       config.d_model,
                                       input_is_parallel=True,
                                       bias=False,
                                       init_method=init_method_normal(config.init_method_std))
        self.output_dropout = nn.Dropout(config.dropout_rate)

        # NOTE: This is a hack for our 130 training head.
        # self.do_dim_trick = config.do_dim_trick
        # if torch.distributed.get_rank() % 5 == 4:
        #     self.head_mask = nn.Parameter(torch.tensor([1.0] * 24 + [0.0, 0.0]), requires_grad=False)
        # else:
        #     self.head_mask = nn.Parameter(torch.tensor([1.0] * 26), requires_grad=False)

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, h_p=n_p*h_i] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head) # [b, s, n_p, h_i]
        tensor = tensor.view(*new_tensor_shape)
        # tensor: [b, n_p, s, h_i]
        return tensor.permute(0, 2, 1, 3)
    
    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        relative_buckets = 0
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # now relative_position is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        relative_postion_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, num_buckets - 1)
        )

        relative_buckets += torch.where(is_small, relative_position, relative_postion_if_large)
        return relative_buckets

    def compute_bias(self, query_length, key_length):
        """ Compute binned relative position bias """
        context_position = torch.arange(query_length, dtype=torch.long)[:, None]
        memory_position = torch.arange(key_length, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
        )
        relative_position_bucket = relative_position_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        key_value_states=None,
        position_bias=None,
        query_length=None,
        past_key_value=None,):

        batch_size, seq_length = hidden_states.shape[:2]

        real_seq_length = seq_length

        if past_key_value is not None:
            assert (
                len(past_key_value) == 2
            ), "past_key_value should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_value)
            )
            real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

        key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

        # hidden_states: [b, s, d_model]
        if key_value_states is not None:
            assert self.is_cross_attn is True
            # mixed_query_layer: [b, s, h_p]
            mixed_query_layer = self.project_q(hidden_states)
            # mixed_key_value_layer: [b, s, 2 * h_p]
            mixed_key_value_layer = self.project_kv(key_value_states)
            (mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_key_value_layer, 2)
        else:
            assert self.is_cross_attn is False
            # hidden_states: [b, s, h]
            mixed_x_layer = self.project(hidden_states)
            # mixed_x_layer: [b, s, 3 * h_p]
            (mixed_query_layer,
             mixed_key_layer,
             mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 3)
            # mixed_***_layer: [b, s, h_p]

        # ***_layer [b, n_p, s, h_i]
        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        if past_key_value is not None and not self.is_cross_attn:
            assert self.is_decoder is True
            # decoder
            # ***_layer: [b, n_p, 1, h_i]
            past_key_layer, past_value_layer = past_key_value
            # past_***_layer: [b, n_p, s-1, h_i]
            key_layer = torch.cat([past_key_layer, key_layer], dim=2)
            value_layer = torch.cat([past_value_layer, value_layer], dim=2)
            # ***_layer: [b, n_p, s_k, h_i]

        # Raw attention scores. [b, n_p, s_q, s_k] compute every head alone
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))

        # NOTE: We follow the implementation of Transformers to remove the scale of attention_acores
        if self.attn_scale:
            attention_scores = attention_scores / math.sqrt(self.hidden_size_per_attention_head)
        
        # relative positional bias
        if position_bias is None:
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.num_attention_heads_per_partition, real_seq_length, key_length), device=attention_scores.device, dtype=attention_scores.dtype
                )
            else:
                position_bias = self.compute_bias(real_seq_length, key_length)
            
            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -seq_length:, :]

        # Apply the attention mask [b, 1, s_q, s_k] and relative position_bias
        # NOTE: 10000 can't be larger otherwise may cause fp16 overflow (max in fp16 = 65504)
        attention_scores = torch.mul(attention_scores, attention_mask) + (-10000.0 * (1.0 - attention_mask) + position_bias)
        # attention_scores = torch.mul(attention_scores, attention_mask) - 10000.0 * (1.0 - attention_mask)
        
        # Attention probabilities. [b, n_p, s_q, s_k]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        with get_cuda_rng_tracker().fork():
            attention_probs = self.attention_dropout(attention_probs)

        # Context layer.
        context_layer = torch.matmul(attention_probs, value_layer)
        # context_layer: [b, n_p, s, h_i]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # context_layer: [b, s, n_p, h_i]
        # if self.do_dim_trick:
        #     head_mask = self.head_mask.view(1, 1, self.head_mask.size(0), 1).expand_as(context_layer)
        #     context_layer = context_layer * head_mask

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer: [b, s, h_p]

        attn_output = self.dense(context_layer)
        # attn_output: [b, s, d_model]
        attn_output = self.output_dropout(attn_output)

        present_key_value_state = (key_layer, value_layer) if self.is_decoder else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        return outputs  # attn_output, present_key_value_state, position_bias


class ParallelSelfAttention(nn.Module):
    def __init__(
        self,
        config: EVAConfig, 
        init_method: Callable,
        is_decoder: bool = False,
        output_layer_init_method: Optional[Callable] = None,
        has_relative_attention_bias: bool = False):
        
        super(ParallelSelfAttention, self).__init__()
        self.self_attn = ParallelAttention(
            config, 
            init_method,
            is_decoder=is_decoder,
            is_cross_attn=False,
            output_layer_init_method=output_layer_init_method, 
            has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        past_key_value=None):

        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.self_attn(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_value=past_key_value,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # add attentions if we output them
        outputs = (hidden_states,) + attention_output[1:]
        return outputs # hidden_states, present_key_value_state, position_bias


class ParallelCrossAttention(nn.Module):
    def __init__(
        self,
        config: EVAConfig,
        init_method: Callable,
        is_decoder: bool = True,
        output_layer_init_method: Optional[Callable] = None):
        
        super(ParallelCrossAttention, self).__init__()

        self.cross_attn = ParallelAttention(
            config,
            init_method,
            is_decoder=is_decoder,
            is_cross_attn=True,
            output_layer_init_method=output_layer_init_method,
            has_relative_attention_bias=False)
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        query_length=None,
        past_key_value=None):

        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.cross_attn(
            normed_hidden_states,
            key_value_states=key_value_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            query_length=query_length,
            past_key_value=past_key_value
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # add attentions if we output them
        outputs = (hidden_states,) + attention_output[1:]
        return outputs # hidden_states, present_key_value_state, position_bias


class ParallelFF(nn.Module):
    def __init__(
        self,
        config: EVAConfig,
        init_method: Callable,
        output_layer_init_method: Callable = None):
        super(ParallelFF, self).__init__()

        self.dense_relu_dense = ParallelDenseReluDense(config, init_method, output_layer_init_method)
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        # hidden_states [b, s, d_model]
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.dense_relu_dense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class ParallelBlock(nn.Module):
    def __init__(
        self, 
        config: EVAConfig,
        init_method: Callable,
        output_layer_init_method: Optional[Callable] = None,
        has_relative_attention_bias: bool = False, 
        is_decoder: bool = False):
        super(ParallelBlock, self).__init__()

        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.is_decoder = is_decoder

        self.self_attn = ParallelSelfAttention(
            config,
            init_method,
            is_decoder=is_decoder,
            output_layer_init_method=output_layer_init_method, 
            has_relative_attention_bias=has_relative_attention_bias)

        if is_decoder:
            self.cross_attn = ParallelCrossAttention(
                config,
                init_method,
                is_decoder=is_decoder,
                output_layer_init_method=output_layer_init_method)

        self.ff = ParallelFF(
            config,
            init_method,
            output_layer_init_method=output_layer_init_method)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        enc_hidden_states=None,
        cross_attention_mask=None,
        enc_dec_position_bias=None,
        past_key_value=None,):

        if past_key_value is not None:
            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attn_outputs = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_value=self_attn_past_key_value,
        )
        hidden_states, present_key_value_state = self_attn_outputs[:2]
        attention_outputs = self_attn_outputs[2:] # position_bias

        # cross attn
        if self.is_decoder:
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attn_outputs = self.cross_attn(
                hidden_states,
                key_value_states=enc_hidden_states,
                attention_mask=cross_attention_mask,
                position_bias=enc_dec_position_bias,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
            )

            hidden_states = cross_attn_outputs[0]
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attn_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attn_outputs[2:]

        hidden_states = self.ff(hidden_states)
        outputs = (hidden_states,)

        outputs = outputs + (present_key_value_state,) + attention_outputs

        # (for encoder) hidden_states, present_key_value_states, self-attention position bias
        # (for decoder) hidden_states, present_key_value_states, self-attention position bias, cross-attention position bias
        return outputs


class ParallelTransformer(nn.Module):
    def __init__(self, config: EVAConfig, word_embeds: VocabParallelEmbedding, role_embeds: nn.Embedding, is_decoder=False, checkpoint_activations=False, checkpoint_num_layers=1):
        super(ParallelTransformer, self).__init__()
        
        self.word_embeds = word_embeds
        self.role_embeds = role_embeds
        # self.position_embeds = nn.Embedding(config.max_position_embeddings, config.d_model)
        # init_method_normal(std=config.init_method_std)(self.position_embeds.weight)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.final_layernorm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.is_decoder = is_decoder

        # output_layer_init_method = None
        # if config.use_scaled_init_for_output_weights:
        #     output_layer_init_method = scaled_init_method(config.init_method_std,
        #                                                   config.num_layers)

        self.blocks = nn.ModuleList(
            [ParallelBlock(
                config,
                init_method=None,
                # unscaled_init_method(sigma=config.init_method_std),
                has_relative_attention_bias=bool(i == 0),
                # output_layer_init_method=output_layer_init_method,
                is_decoder=is_decoder) for i in range(config.num_decoder_layers if self.is_decoder else config.num_layers)]
        )

        if deepspeed.checkpointing.is_configured():
            global get_cuda_rng_tracker, checkpoint
            get_cuda_rng_tracker = deepspeed.checkpointing.get_cuda_rng_tracker
            checkpoint = deepspeed.checkpointing.checkpoint

    def forward(
        self,
        input_ids=None,
        role_ids=None,
        attention_mask=None,
        cross_attention_mask=None,
        enc_hidden_states=None,
        past_key_values=None,):
        
        inputs_embeds = self.word_embeds(input_ids)
        if role_ids is not None:
            role_embeds = self.role_embeds(role_ids)
            # add role embeddings
            inputs_embeds = inputs_embeds + role_embeds
        
        # remove abstract position ids
        # pos_embeds = self.position_embeds(position_ids)
        # inputs_embeds = inputs_embeds + pos_embeds


        hidden_states = self.dropout(inputs_embeds)
        position_bias = None
        enc_dec_position_bias = None
        present_key_value_states = []

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.blocks)

        # NOTE: check implementation: checkpoint_activations

        def custom(start, end):
            def custom_forward(*inputs):                
                layer_modules_ = self.blocks[start:end]
                past_key_values_ = past_key_values[start:end]
                present_key_values_ = []
                position_bias_, enc_dec_position_bias_ = None, None

                hidden_states_ = inputs[0]
                if len(inputs) > 2:
                    position_bias_ = inputs[1]
                if len(inputs) > 3:
                    enc_dec_position_bias_ = inputs[2]
                
                if enc_hidden_states is not None:
                    enc_hidden_states_ = inputs[-1]
                else:
                    enc_hidden_states_ = None

                for layer_, past_key_value_ in zip(layer_modules_, past_key_values_):
                    layer_outputs_ = layer_(hidden_states_,
                                            attention_mask,
                                            position_bias_,
                                            enc_hidden_states_,
                                            cross_attention_mask,
                                            enc_dec_position_bias_,
                                            past_key_value=past_key_value_)
                    
                    hidden_states_, present_key_value_ = layer_outputs_[:2]

                    position_bias_ = layer_outputs_[2]
                    if self.is_decoder and enc_hidden_states is not None:
                        enc_dec_position_bias_ = layer_outputs_[3]

                outputs_ = (hidden_states_,)
                if position_bias_ is not None:
                    outputs_ += (position_bias_,)
                if enc_dec_position_bias_ is not None:
                    outputs_ += (enc_dec_position_bias_,)

                return outputs_
            
            return custom_forward

        if self.checkpoint_activations:
            l = 0
            num_layers = len(self.blocks)
            chunk_length = self.checkpoint_num_layers
            while l < num_layers:
                arg_list = (hidden_states,)
                if position_bias is not None:
                    arg_list += (position_bias,)
                if enc_dec_position_bias is not None:
                    arg_list += (enc_dec_position_bias,)
                
                if enc_hidden_states is not None:
                    arg_list += (enc_hidden_states,)
                    tmp_outputs = checkpoint(custom(l, l+chunk_length), *arg_list)
                else:
                    arg_list += (attention_mask,)
                    tmp_outputs = checkpoint(custom(l, l+chunk_length), *arg_list)
                
                hidden_states = tmp_outputs[0]
                if len(tmp_outputs) > 1:
                    position_bias = tmp_outputs[1]
                if len(tmp_outputs) > 2:
                    enc_dec_position_bias = tmp_outputs[2]

                # NOTE: we didn't consider present_key_value_states and all_hidden_states
                present_key_value_states.extend([None] * chunk_length)
                
                l += chunk_length
        else:
            for i, (layer_module, past_key_value) in enumerate(zip(self.blocks, past_key_values)):

                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_bias=position_bias,
                    enc_hidden_states=enc_hidden_states,
                    cross_attention_mask=cross_attention_mask,
                    enc_dec_position_bias=enc_dec_position_bias,
                    past_key_value=past_key_value
                )
                # layer_outputs is a tuple with:
                # hidden-states, key-value-states, self-attention position bias, cross-attention position bias
                hidden_states, present_key_value_state = layer_outputs[:2]
                
                position_bias = layer_outputs[2]
                if self.is_decoder and enc_hidden_states is not None:
                    enc_dec_position_bias = layer_outputs[3]
                
                present_key_value_states.append(present_key_value_state)
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights),
                # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                # position_bias = layer_outputs[2]

        hidden_states = self.final_layernorm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        outputs = {
            "last_hidden_state": hidden_states,
            "past_key_values": present_key_value_states,
            "hidden_states": None,
            "attentions": None,
            "cross_attentions": None
        }

        return outputs
