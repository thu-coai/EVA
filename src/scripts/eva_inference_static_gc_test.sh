#! /bin/bash

WORKING_DIR=/root/eva-origin/

# Change for multinode config
MP_SIZE=1

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=4

DATA_PATH="${WORKING_DIR}/data/kdconv"

CONFIG_PATH="${WORKING_DIR}/src/configs/model/eva2.0_model_config.json"
CKPT_PATH="${WORKING_DIR}/checkpoints/eva2.0"

LR=${2-0.001}
WM=${3-0.01}
GRAD_ACC=${4-1}

SAVE_PATH="${WORKING_DIR}/results/test_eva_infer_gc/"
# SAVE_PATH="${WORKING_DIR}/results/test/"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/src/configs/deepspeed/eva_ds_config.json"
TOKENIZER_PATH="${WORKING_DIR}/bpe_dialog_new"

BATCH_SIZE=32
TRAIN_ITER=-1
EPOCHS=3

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
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --warmup ${WM}"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --do-eval"
OPTS+=" --eval-generation"


export NCCL_DEBUG=WARN

CMD="python3 -m torch.distributed.launch --master_port 1234 --nproc_per_node ${NUM_GPUS_PER_WORKER} ${WORKING_DIR}/src/eva_finetune.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
