#! /bin/bash

WORKING_DIR=/home/leichangsong/eva-huggingface

MP_SIZE=1

NUM_GPUS_PER_WORKER=1

DATA_PATH="/home/leichangsong/transformers/data"

CONFIG_PATH="${WORKING_DIR}/model_config/eva2.0_model_config.json"
CKPT_PATH="/home/leichangsong/another.pt"

SAVE_PATH="${WORKING_DIR}/results/inference_static/"
LOG_FILE="${SAVE_PATH}/log.txt"
TOKENIZER_PATH="${WORKING_DIR}/vocab"
RULE_PATH="${WORKING_DIR}/rules"
BATCH_SIZE=4

TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=0
TOPP=0.9
NUM_BEAMS=4

ENC_LEN=128
DEC_LEN=128


OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
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
OPTS+=" --do-eval"
OPTS+=" --eval-generation"
OPTS+=" --num-beams ${NUM_BEAMS}"
OPTS+=" --temperature ${TEMP}"
OPTS+=" --top_k ${TOPK}"
OPTS+=" --rule-path ${RULE_PATH}"
OPTS+=" --top_p ${TOPP}"
PTS+=" --length-penalty 1.6"
OPTS+=" --repetition-penalty 1.6"


CMD="python ../eva_evaluate.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
