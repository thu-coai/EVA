import torch

ckpt_eva = torch.load('/home/COAI/EVA/checkpoints/eva2.0_large/1/mp_rank_00_model_states.pt', map_location="cpu")
ckpt_eva = ckpt_eva["module"]
ckpt_hf = {}

ckpt_hf['shared.weight'] = ckpt_eva['word_embeds.weight']
if "role_embeds.weight" in ckpt_hf:
    ckpt_hf['role_embeds.weight'] = ckpt_eva['role_embeds.weight']
if "encoder.role_embeds.weight" in ckpt_hf:
    ckpt_hf['encoder.role_embeds.weight'] = ckpt_eva['encoder.role_embeds.weight']
if "decoder.role_embeds.weight" in ckpt_hf:
    ckpt_hf['decoder.role_embeds.weight'] = ckpt_eva['decoder.role_embeds.weight']
ckpt_hf['lm_head.weight'] = ckpt_eva['lm_head.weight']
ckpt_hf['encoder.embed_tokens.weight'] = ckpt_eva['encoder.word_embeds.weight']
ckpt_hf['encoder.final_layer_norm.weight'] = ckpt_eva['encoder.final_layernorm.weight']
ckpt_hf['encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'] = ckpt_eva['encoder.blocks.0.self_attn.self_attn.relative_attention_bias.weight']
ckpt_hf['decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight'] = ckpt_eva['decoder.blocks.0.self_attn.self_attn.relative_attention_bias.weight']
ckpt_hf['decoder.embed_tokens.weight'] = ckpt_eva['decoder.word_embeds.weight']
ckpt_hf['decoder.final_layer_norm.weight'] = ckpt_eva['decoder.final_layernorm.weight']

layers = 24

for i in range(layers):
    attn_proj = ckpt_eva[f'encoder.blocks.{i}.self_attn.self_attn.project.weight']
    assert attn_proj.size(0) % 3 == 0
    d_model = attn_proj.size(0) // 3
    ckpt_hf[f'encoder.block.{i}.layer.0.SelfAttention.q.weight'] = attn_proj[:d_model,:]
    ckpt_hf[f'encoder.block.{i}.layer.0.SelfAttention.k.weight'] = attn_proj[d_model:2*d_model,:]
    ckpt_hf[f'encoder.block.{i}.layer.0.SelfAttention.v.weight'] = attn_proj[2*d_model:,:]
    
    ckpt_hf[f'encoder.block.{i}.layer.0.SelfAttention.o.weight'] = ckpt_eva[f'encoder.blocks.{i}.self_attn.self_attn.dense.weight']
    ckpt_hf[f'encoder.block.{i}.layer.0.layer_norm.weight'] = ckpt_eva[f'encoder.blocks.{i}.self_attn.layer_norm.weight']
    ckpt_hf[f'encoder.block.{i}.layer.1.DenseReluDense.wi_0.weight'] = ckpt_eva[f'encoder.blocks.{i}.ff.dense_relu_dense.wi_0.weight']
    ckpt_hf[f'encoder.block.{i}.layer.1.DenseReluDense.wi_1.weight'] = ckpt_eva[f'encoder.blocks.{i}.ff.dense_relu_dense.wi_1.weight']
    ckpt_hf[f'encoder.block.{i}.layer.1.DenseReluDense.wo.weight'] = ckpt_eva[f'encoder.blocks.{i}.ff.dense_relu_dense.wo.weight']
    ckpt_hf[f'encoder.block.{i}.layer.1.layer_norm.weight'] = ckpt_eva[f'encoder.blocks.{i}.ff.layer_norm.weight']


for i in range(layers):
    attn_proj = ckpt_eva[f'decoder.blocks.{i}.self_attn.self_attn.project.weight']
    assert attn_proj.size(0) % 3 == 0
    d_model = attn_proj.size(0) // 3
    ckpt_hf[f'decoder.block.{i}.layer.0.SelfAttention.q.weight'] = attn_proj[:d_model,:]
    ckpt_hf[f'decoder.block.{i}.layer.0.SelfAttention.k.weight'] = attn_proj[d_model:2*d_model,:]
    ckpt_hf[f'decoder.block.{i}.layer.0.SelfAttention.v.weight'] = attn_proj[2*d_model:,:]

    ckpt_hf[f'decoder.block.{i}.layer.0.SelfAttention.o.weight'] = ckpt_eva[f"decoder.blocks.{i}.self_attn.self_attn.dense.weight"]
    ckpt_hf[f'decoder.block.{i}.layer.0.layer_norm.weight'] = ckpt_eva[f'decoder.blocks.{i}.self_attn.layer_norm.weight']

    ckpt_hf[f'decoder.block.{i}.layer.1.EncDecAttention.q.weight'] = ckpt_eva[f'decoder.blocks.{i}.cross_attn.cross_attn.project_q.weight']
    cross_attn_proj = ckpt_eva[f'decoder.blocks.{i}.cross_attn.cross_attn.project_kv.weight']
    assert cross_attn_proj.size(0) % 2 == 0
    ckpt_hf[f'decoder.block.{i}.layer.1.EncDecAttention.k.weight'] = cross_attn_proj[:d_model,:]
    ckpt_hf[f'decoder.block.{i}.layer.1.EncDecAttention.v.weight'] = cross_attn_proj[d_model:,:]
    ckpt_hf[f'decoder.block.{i}.layer.1.EncDecAttention.o.weight'] = ckpt_eva[f'decoder.blocks.{i}.cross_attn.cross_attn.dense.weight']
    ckpt_hf[f'decoder.block.{i}.layer.1.layer_norm.weight'] = ckpt_eva[f'decoder.blocks.{i}.cross_attn.layer_norm.weight']

    ckpt_hf[f'decoder.block.{i}.layer.2.DenseReluDense.wi_0.weight'] = ckpt_eva[f'decoder.blocks.{i}.ff.dense_relu_dense.wi_0.weight']
    ckpt_hf[f'decoder.block.{i}.layer.2.DenseReluDense.wi_1.weight'] = ckpt_eva[f'decoder.blocks.{i}.ff.dense_relu_dense.wi_1.weight']
    ckpt_hf[f'decoder.block.{i}.layer.2.DenseReluDense.wo.weight'] = ckpt_eva[f'decoder.blocks.{i}.ff.dense_relu_dense.wo.weight']
    ckpt_hf[f'decoder.block.{i}.layer.2.layer_norm.weight'] = ckpt_eva[f'decoder.blocks.{i}.ff.layer_norm.weight']


torch.save(ckpt_hf,'/home/COAI/EVA/checkpoints/eva2.0_large-hf/pytorch_model.bin')