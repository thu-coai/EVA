#! /bin/bash

WORKING_DIR=/path-to-eva/EVA/

# Change for multinode config
MP_SIZE=1

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=1

CONFIG_PATH="${WORKING_DIR}/src/configs/model/eva_model_config.json"

CKPT_PATH="/path-to-checkpoint/eva/"
DS_CONFIG="${WORKING_DIR}/src/configs/deepspeed/eva_ds_config.json"
TOKENIZER_PATH="${WORKING_DIR}/src/bpe_dialog_new"

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

CMD="deepspeed --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --master_port 4586 --master_addr localhost ${WORKING_DIR}/src/eva_interactive.py ${OPTS}"

echo ${CMD}
${CMD}
