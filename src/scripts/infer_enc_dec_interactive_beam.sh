#! /bin/bash

WORKING_DIR=/dataset/f1d6ea5b/gyx-eva/eva-origin/

# Change for multinode config
MP_SIZE=1

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1

CONFIG_PATH="${WORKING_DIR}/src/configs/model/eva_model_config_attn_scale.json"

# CKPT_PATH="/dataset/f1d6ea5b/gyx-eva/eva2/results/finetune_with_pretrain_0.00005_128_1/17500"
# CKPT_PATH="/dataset/f1d6ea5b/gyx-eva/eva2/results/finetune_with_pretrain_0.00005_128_1/"
# CKPT_PATH="${WORKING_DIR}/checkpoints/eva"
CKPT_PATH="/dataset/f1d6ea5b/gyx-eva/eva2/results/new_data_scale_1103/120000"
# CKPT_PATH="/dataset/f1d6ea5b/gyx-eva/eva2/results/finetune_with_pretrain_0.00005_32_1_esc_9_13"
# CKPT_PATH="/dataset/f1d6ea5b/gyx-eva/eva2/results/finetune_with_pretrain_0.00005_32_1_esc_blender_9_15/8000"
# CKPT_PATH="/dataset/f1d6ea5b/gyx-eva/eva2/results/finetune_with_pretrain_0.00005_32_1_12G_9_7/60000/"
RANKER_CKPT_PATH="${WORKING_DIR}/checkpoints/ranker/pytorch_model.bin"
# RANKER_CKPT_PATH="/dataset/f1d6ea5b/wenjiaxin/ruber-pytorch/results/chinese-roberta-wwm-ext_dialog_0920_nlpcc_retrival/lightning_logs/version_0/checkpoints/converted_epoch=1-step=31249.ckpt"
RANKER_CONFIG_PATH="/${WORKING_DIR}//checkpoints/ranker"
# CKPT_PATH="/dataset/f1d6ea5b/gyx-eva/eva2//results/enc_dec_eva-depth-12-48-lr0.0005-wm0.01/"
# CKPT_PATH="/dataset/f1d6ea5b/gyx-eva/eva2/results/finetune_with_pretrain_0.00005_128_1/"
DS_CONFIG="${WORKING_DIR}/src/configs/deepspeed/eva_ds_config.json"
TOKENIZER_PATH="${WORKING_DIR}/bpe_dialog_new"
HOST_FILE="${WORKING_DIR}/src/configs/host_files/hostfile-s1"

TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=0
TOPP=0.9
NUM_BEAMS=4


OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --ranker-load ${RANKER_CKPT_PATH}"
OPTS+=" --ranker-config ${RANKER_CONFIG_PATH}"
OPTS+=" --distributed-backend nccl"
OPTS+=" --no-load-optim"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --temperature ${TEMP}"
OPTS+=" --top_k ${TOPK}"
OPTS+=" --top_p ${TOPP}"
OPTS+=" --num-beams ${NUM_BEAMS}"
OPTS+=" --length-penalty 1.6"
OPTS+=" --repetition-penalty 1.6"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
# OPTS+=" --rerank"
# OPTS+=" --rerank_num 5"
# OPTS+=" --human_rules"

CMD="/opt/conda/bin/deepspeed -i cgpt-s1:6 --master_port 1345 --hostfile ${HOST_FILE} ${WORKING_DIR}/src/eva_interactive.py ${OPTS}"

echo ${CMD}
${CMD}
