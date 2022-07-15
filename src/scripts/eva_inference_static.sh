#! /bin/bash

WORKING_DIR=/home/COAI/EVA

NUM_GPUS_PER_WORKER=1

DATA_PATH="${WORKING_DIR}/data/kdconv"
CKPT_PATH="${WORKING_DIR}/checkpoints/eva2.0-hf"

SAVE_PATH="${WORKING_DIR}/results/inference_static/"
LOG_FILE="${SAVE_PATH}/log.txt"
BATCH_SIZE=8

TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=0
TOPP=0.9
NUM_BEAMS=1

ENC_LEN=128
DEC_LEN=128


OPTS=""
OPTS+=" --eval-batch-size ${BATCH_SIZE}"
OPTS+=" --enc-seq-length ${ENC_LEN}"
OPTS+=" --dec-seq-length ${DEC_LEN}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --distributed-backend nccl"
OPTS+=" --eval-generation"
OPTS+=" --num-beams ${NUM_BEAMS}"
OPTS+=" --temperature ${TEMP}"
OPTS+=" --top_k ${TOPK}"
OPTS+=" --top_p ${TOPP}"
OPTS+=" --length-penalty 1.6"
OPTS+=" --repetition-penalty 1.6"


CMD="torchrun --master_port 1234 --nproc_per_node ${NUM_GPUS_PER_WORKER} ${WORKING_DIR}/src/eva_evaluate.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
