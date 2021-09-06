#! /bin/bash

WORKING_DIR=/dataset/f1d6ea5b/gyx-eva/eva-origin/

# Change for multinode config
MP_SIZE=1

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1

CONFIG_PATH="${WORKING_DIR}/src/configs/model/eva_model_config.json"

# CKPT_PATH="/dataset/f1d6ea5b/gyx-eva/eva2/results/enc_dec_eva-5-17/360000"
# CKPT_PATH="/dataset/f1d6ea5b/gyx-eva/eva2//results/enc_dec_eva-depth-12-48-lr0.0005-wm0.01/"
CKPT_PATH="/dataset/f1d6ea5b/gyx-eva/eva2/results/finetune_with_pretrain_0.00005_128_1/"
DS_CONFIG="${WORKING_DIR}/src/configs/deepspeed/eva_ds_config.json"
TOKENIZER_PATH="${WORKING_DIR}/bpe_dialog_new"
HOST_FILE="${WORKING_DIR}/src/configs/host_files/hostfile"

TEMP=0.7
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=0
TOPP=0.9

OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --distributed-backend nccl"
OPTS+=" --no-load-optim"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --temperature ${TEMP}"
OPTS+=" --top_k ${TOPK}"
OPTS+=" --top_p ${TOPP}"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"

CMD="/opt/conda/bin/deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --master_port 4586 --hostfile ${HOST_FILE} ${WORKING_DIR}/src/eva_interactive.py ${OPTS}"

echo ${CMD}
${CMD}
