# coding=utf-8

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import CrossEntropyLoss
from typing import Callable, Optional
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput
from transformers.utils import logging
from .configuration_eva import EVAConfig
from transformers.file_utils import DUMMY_INPUTS,DUMMY_MASK,is_torch_fx_proxy


logger = logging.get_logger(__name__)


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states


class DenseGatedGeluDense(nn.Module):
    def __init__(self, config: EVAConfig):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = F.gelu

    def forward(self, hidden_states):
        hidden_gelu = self.gelu_act(self.wi_0(hidden_states))
        hidden_linear = self.wi_1(hidden_states)
        hidden_states = hidden_gelu * hidden_linear
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.wo(hidden_states)
        return hidden_states


class Attention(nn.Module):
    def __init__(
        self,
        config: EVAConfig, 
        init_method: Callable,
        is_decoder: bool = False,
        is_cross_attn: bool = False,
        output_layer_init_method: Optional[Callable] = None,
        has_relative_attention_bias: bool = False):
        super(Attention, self).__init__()

        self.is_decoder = is_decoder
        self.is_cross_attn = is_cross_attn
        self.attn_scale = config.attn_scale

        self.has_relative_attention_bias = has_relative_attention_bias
        self.relative_attention_num_buckets = config.relative_attention_num_buckets

        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        
        self.d_attn_out = config.d_kv * config.num_heads # h
        self.hidden_size_per_partition = self.d_attn_out
        
        # Per attention head and per partition values.
        self.num_heads = config.num_heads
        self.num_attention_heads_per_partition = config.num_heads
        # self.hidden_size_per_partition = divide(self.d_attn_out, world_size) # h_p
        self.hidden_size_per_attention_head = config.d_kv # h_i
        # self.num_attention_heads_per_partition = divide(config.num_heads, world_size) # n_p
        self.q = nn.Linear(config.d_model, self.d_attn_out, bias=False)
        self.k = nn.Linear(config.d_model, self.d_attn_out, bias=False)
        self.v = nn.Linear(config.d_model, self.d_attn_out, bias=False)
        self.o = nn.Linear(self.d_attn_out, config.d_model, bias=False)
        
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.num_heads)  
        
        # Dropout. Note that for a single iteration, this layer will generate
        # on average it should not be partition dependent.
        # self.attention_dropout = nn.Dropout(config.dropout_rate)

        # Output.
        self.dropout = nn.Dropout(config.dropout_rate)

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
        past_key_value=None,
        use_cache=False,
        output_attentions=False,):

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

        if key_value_states is not None:
            assert self.is_cross_attn is True
            # mixed_query_layer: [b, s, h_p]
            mixed_query_layer = self.q(hidden_states)
            # mixed_key_value_layer: [b, s, 2 * h_p]
            mixed_key_layer = self.k(key_value_states)
            mixed_value_layer = self.v(key_value_states)
        else:
            assert self.is_cross_attn is False
            # hidden_states: [b, s, h]
            mixed_query_layer = self.q(hidden_states)
            mixed_key_layer = self.k(hidden_states)
            mixed_value_layer = self.v(hidden_states)
            # mixed_***_layer: [b, s, h_p]


        query_layer = self._transpose_for_scores(mixed_query_layer)
        key_layer = self._transpose_for_scores(mixed_key_layer)
        value_layer = self._transpose_for_scores(mixed_value_layer)

        if past_key_value is not None and not self.is_cross_attn:
            assert self.is_decoder is True
            # decoder
            # ***_layer: [b, n_p, 1, h_i]
            past_key_layer, past_value_layer = past_key_value
            # past_***_layer: [b, n_p, s-1, h_i]
            key_layer = torch.cat([past_key_layer, key_layer.to(past_key_layer.device)], dim=2)
            value_layer = torch.cat([past_value_layer, value_layer.to(past_value_layer.device)], dim=2)
            # ***_layer: [b, n_p, s_k, h_i]

        attention_scores = torch.matmul(query_layer.to(key_layer.device),
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
        attention_scores = attention_scores + attention_mask + position_bias
        
        # Attention probabilities. [b, n_p, s_q, s_k]
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Context layer.
        context_layer = torch.matmul(attention_probs.to(value_layer.device), value_layer)
        # context_layer: [b, n_p, s, h_i]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # context_layer: [b, s, n_p, h_i]
        # if self.do_dim_trick:
        #     head_mask = self.head_mask.view(1, 1, self.head_mask.size(0), 1).expand_as(context_layer)
        #     context_layer = context_layer * head_mask

        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)
        # context_layer: [b, s, h_p]

        attn_output = self.o(context_layer.to(self.o.weight.device))
        # attn_output: [b, s, d_model]
        attn_output = self.dropout(attn_output)

        present_key_value_state = torch.stack((key_layer, value_layer), dim=0) if (self.is_decoder and use_cache) else None
        outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

        if output_attentions:
            outputs = outputs + (attention_probs,)

        return outputs  # attn_output, present_key_value_state, position_bias, attention_probs


class SelfAttention(nn.Module):
    def __init__(
        self,
        config: EVAConfig, 
        init_method: Callable,
        is_decoder: bool = False,
        output_layer_init_method: Optional[Callable] = None,
        has_relative_attention_bias: bool = False):
        
        super(SelfAttention, self).__init__()
        self.SelfAttention = Attention(
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
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):

        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            normed_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # add attentions if we output them
        outputs = (hidden_states,) + attention_output[1:]
        return outputs # hidden_states, present_key_value_state, position_bias, (attention_probs)


class CrossAttention(nn.Module):
    def __init__(
        self,
        config: EVAConfig,
        init_method: Callable,
        is_decoder: bool = True,
        output_layer_init_method: Optional[Callable] = None):
        
        super(CrossAttention, self).__init__()

        self.EncDecAttention = Attention(
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
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):

        normed_hidden_states = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            normed_hidden_states,
            key_value_states=key_value_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            query_length=query_length,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # add attentions if we output them
        outputs = (hidden_states,) + attention_output[1:]
        return outputs # hidden_states, present_key_value_state, position_bias, (attention_probs)


class FF(nn.Module):
    def __init__(
        self,
        config: EVAConfig):
        super(FF, self).__init__()
        self.DenseReluDense = DenseGatedGeluDense(config)
        self.layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        # hidden_states [b, s, d_model]
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class Block(nn.Module):
    def __init__(
        self, 
        config: EVAConfig,
        init_method: Callable,
        output_layer_init_method: Optional[Callable] = None,
        has_relative_attention_bias: bool = False, 
        is_decoder: bool = False):
        super(Block, self).__init__()

        if output_layer_init_method is None:
            output_layer_init_method = init_method

        self.is_decoder = is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(SelfAttention(
            config,
            init_method,
            is_decoder=is_decoder,
            output_layer_init_method=output_layer_init_method, 
            has_relative_attention_bias=has_relative_attention_bias))

        if self.is_decoder:
            self.layer.append(CrossAttention(
                config,
                init_method,
                is_decoder=is_decoder,
                output_layer_init_method=output_layer_init_method))

        self.layer.append(FF(config))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        enc_hidden_states=None,
        cross_attention_mask=None,
        enc_dec_position_bias=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):

        if past_key_value is not None:
            self_attn_past_key_value = past_key_value[0]
            cross_attn_past_key_value = past_key_value[1]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attn_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, self_attn_present_key_value = self_attn_outputs[:2]
        attn_outputs = self_attn_outputs[2:]
        if self_attn_present_key_value is not None:
            present_key_value = (self_attn_present_key_value,)
        else:
            present_key_value = None

        if self.is_decoder:
            if self_attn_present_key_value is not None:
                query_length = self_attn_present_key_value[0].shape[2]
            else:
                query_length = None

            cross_attn_outputs = self.layer[1](
                hidden_states,
                key_value_states=enc_hidden_states,
                attention_mask=cross_attention_mask,
                position_bias=enc_dec_position_bias,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states, cross_attn_present_key_value = cross_attn_outputs[:2]
            if present_key_value is not None:
                present_key_value += (cross_attn_present_key_value,)
            attn_outputs = attn_outputs + cross_attn_outputs[2:]

        hidden_states = self.layer[-1](hidden_states)
        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value,) + attn_outputs # hidden_states, present_key_value, self_attn_position_bias, self_attn_probs, cross_attn_position_bias, cross_attn_probs
        else:
            outputs = outputs + attn_outputs

        return outputs


class EVAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = EVAConfig
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """Initialize the weights"""
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, EVAModel):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, DenseGatedGeluDense):
            module.wi_0.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            module.wi_1.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            key_value_proj_dim = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * key_value_proj_dim) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * key_value_proj_dim) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Attention, Transformer)):
            module.gradient_checkpointing = value

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        if is_torch_fx_proxy(input_ids):
            # Item assignment is not supported natively for proxies.
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that `shifted_input_ids` has only positive values"

        return shifted_input_ids


class Transformer(EVAPreTrainedModel):
    def __init__(self, config: EVAConfig, word_embeds: nn.Embedding, role_embeds: nn.Embedding, is_decoder=False, checkpoint_activations=False, checkpoint_num_layers=1):
        super(Transformer, self).__init__(config)
        
        self.config = config
        self.embed_tokens = word_embeds
        self.role_embeds = role_embeds
        self.dropout = nn.Dropout(config.dropout_rate)
        self.final_layer_norm = LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.is_decoder = is_decoder

        self.block = nn.ModuleList(
            [Block(
                config,
                init_method=None,
                has_relative_attention_bias=bool(i == 0),
                is_decoder=is_decoder) for i in range(config.num_layers)]
        )
        self.post_init()
        
    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings
    
    def forward(
        self,
        input_ids=None,
        role_ids=None,
        attention_mask=None,
        cross_attention_mask=None,
        enc_hidden_states=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        inputs_embeds = self.embed_tokens(input_ids.to(self.embed_tokens.weight.device))
        if role_ids is not None:
            role_embeds = self.role_embeds(role_ids)
            # add role embeddings
            inputs_embeds = inputs_embeds + role_embeds
        
        input_shape = input_ids.size()
        batch_size, seq_length = input_shape
        mask_seq_length = past_key_values[0][0].shape[3] + seq_length if past_key_values is not None else seq_length
        
        if use_cache:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        
        if attention_mask.dim() == 2 and self.is_decoder and past_key_values is None:
            attention_mask = attention_mask.view(
                batch_size, mask_seq_length, 1) * attention_mask.view(batch_size, 1, mask_seq_length)
            attention_mask = torch.tril(attention_mask)
            
        if self.is_decoder and cross_attention_mask is None and enc_hidden_states is not None:
            encoder_seq_length = enc_hidden_states.shape[1]
            cross_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )
        
        attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        if self.is_decoder and enc_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = enc_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if cross_attention_mask is None:
                cross_attention_mask = torch.ones(encoder_hidden_shape, device=inputs_embeds.device)
            cross_attention_mask = self.invert_attention_mask(cross_attention_mask)
        else:
            cross_attention_mask = None

        hidden_states = self.dropout(inputs_embeds)
        position_bias = None
        enc_dec_position_bias = None
        present_key_value_states = [] if use_cache else None

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        all_hidden_states = [] if output_hidden_states else None
        all_self_attention_probs = [] if output_attentions else None
        all_cross_attention_probs = [] if (output_attentions and self.is_decoder) else None

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):

            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                enc_hidden_states=enc_hidden_states,
                cross_attention_mask=cross_attention_mask,
                enc_dec_position_bias=enc_dec_position_bias,
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, self-attention position bias, self_attn_probs, cross-attention position bias, cross_attn_probs
            
            if not use_cache:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]
            
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs[2]
            if self.is_decoder and enc_hidden_states is not None:
                enc_dec_position_bias = layer_outputs[4 if output_attentions else 3]
            
            if use_cache:
                present_key_value_states.append(present_key_value_state)
                
            if output_attentions:
                all_self_attention_probs.append(layer_outputs[3])
                if self.is_decoder:
                    all_cross_attention_probs.append(layer_outputs[5])

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_self_attention_probs,
                    all_cross_attention_probs,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attention_probs,
            cross_attentions=all_cross_attention_probs,
        )


class EVAModel(EVAPreTrainedModel):
    def __init__(
        self,
        config: EVAConfig,
        checkpoint_activations=False,
        checkpoint_num_layers=1):
        
        super(EVAModel, self).__init__(config)
        if config.vocab_size is None:
            raise RuntimeError("Should set vocab size")

        self.config = config
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        self.role_embeds = nn.Embedding(2, config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.encoder = Transformer(self.config, word_embeds=self.shared, role_embeds=self.role_embeds, is_decoder=False,
                                               checkpoint_activations=checkpoint_activations, checkpoint_num_layers=checkpoint_num_layers)
        self.decoder = Transformer(self.config, word_embeds=self.shared, role_embeds=self.role_embeds, is_decoder=True,
                                               checkpoint_activations=checkpoint_activations, checkpoint_num_layers=checkpoint_num_layers)
        self.post_init()


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self, 
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        encoder_attention_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        only_encoder=False,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
        
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        encoder_hidden_states = encoder_outputs[0]

        if only_encoder:
            return encoder_outputs

        if labels is not None and decoder_input_ids is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            cross_attention_mask=encoder_attention_mask,
            enc_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        lm_logits = self.lm_head(decoder_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output
        
        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
        
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        encoder_outputs=None,
        use_cache=None,
        **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
        
    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        # past_ = num_layer * 2 * (2, beam_size, 32, prefix_len, 64) first 2: self/cross attention, second 2: key/value
        reordered_decoder_past = [[torch.index_select(layer_past_type, 1, beam_idx) for layer_past_type in layer_past] for layer_past in past]
        
        return reordered_decoder_past


def enc_dec_get_params_for_weight_decay_optimization(module):

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (nn.LayerNorm, LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params
