#! /bin/bash

WORKING_DIR=/home/coai/EVA/

MP_SIZE=1

NUM_GPUS_PER_WORKER=8

DATA_PATH="${WORKING_DIR}/data/kdconv"

CONFIG_PATH="${WORKING_DIR}/src/configs/model/eva2.0_model_config.json"
CKPT_PATH="${WORKING_DIR}/checkpoints/eva2.0"

SAVE_PATH="${WORKING_DIR}/results/inference_static/"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/src/configs/deepspeed/eva_ds_config.json"
TOKENIZER_PATH="${WORKING_DIR}/bpe_dialog_new"

BATCH_SIZE=32

TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=0
TOPP=0.9
NUM_BEAMS=1

ENC_LEN=128
DEC_LEN=128


OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --eval-batch-size ${BATCH_SIZE}"
OPTS+=" --enc-seq-length ${ENC_LEN}"
OPTS+=" --dec-seq-length ${DEC_LEN}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --no_load_strict"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --distributed-backend nccl"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --do-eval"
OPTS+=" --eval-generation"
OPTS+=" --num-beams ${NUM_BEAMS}"
OPTS+=" --temperature ${TEMP}"
OPTS+=" --top_k ${TOPK}"
OPTS+=" --top_p ${TOPP}"

CMD="torchrun --master_port 1234 --nproc_per_node ${NUM_GPUS_PER_WORKER} ${WORKING_DIR}/src/eva_finetune.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
