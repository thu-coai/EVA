#! /bin/bash

WORKING_DIR=/root/guyuxian/eva-origin/

# Change for multinode config
MP_SIZE=1

NUM_GPUS_PER_WORKER=8

DATA_PATH="/dataset/f1d6ea5b/gyx/data/duconv"

CONFIG_PATH="${WORKING_DIR}/src/configs/model/eva2.0_model_config.json"
CKPT_PATH="${WORKING_DIR}/checkpoints/eva2.0"

LR=${2-0.0001}
WM=${3-0.01}
GRAD_ACC=${4-1}

SAVE_PATH="${WORKING_DIR}/results/test_eva_finetune_wm_0.01_2_param/"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/src/configs/deepspeed/eva_ds_config.json"
TOKENIZER_PATH="${WORKING_DIR}/bpe_dialog_new"

BATCH_SIZE=32
TRAIN_ITER=-1
EPOCHS=5

ENC_LEN=128
DEC_LEN=128


OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OTPS+=" --epochs ${EPOCHS}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --enc-seq-length ${ENC_LEN}"
OPTS+=" --dec-seq-length ${DEC_LEN}"
OPTS+=" --train-iters ${TRAIN_ITER}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --no-load-optim"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --warmup ${WM}"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --no-save-optim"
OPTS+=" --no-save-rng"
OPTS+=" --eval-interval 500"
OPTS+=" --log-interval 100"
OPTS+=" --save-interval 500"
OPTS+=" --checkpoint-activations"
OPTS+=" --deepspeed-activation-checkpointing"
OPTS+=" --fp16"
OPTS+=" --deepspeed"
OPTS+=" --deepspeed_config ${DS_CONFIG}"
OPTS+=" --do-train"
OPTS+=" --do-valid"
OPTS+=" --train-ratio 1"

CMD="python3 -m torch.distributed.launch --master_port 1234 --nproc_per_node ${NUM_GPUS_PER_WORKER} ${WORKING_DIR}/src/eva_finetune.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
