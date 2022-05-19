#! /bin/bash

WORKING_DIR=/home/coai/EVA/

MP_SIZE=1 # the model parallel size

NUM_GPUS_PER_WORKER=4 # number of gpus used on one node

DATA_PATH="${WORKING_DIR}/data/kdconv" # path of the directory of the dataset

CONFIG_PATH="${WORKING_DIR}/src/configs/model/eva2.0_model_config.json"
CKPT_PATH="${WORKING_DIR}/checkpoints/eva2.0"

LR=${2-0.0001} # learning rate
WM=${3-0.01} # ratio of warmup steps
GRAD_ACC=${4-1} # gradient accumulation steps

SAVE_PATH="${WORKING_DIR}/results/finetune/"
LOG_FILE="${SAVE_PATH}/log.txt"
DS_CONFIG="${WORKING_DIR}/src/configs/deepspeed/eva_ds_config.json" # config of deepspeed
TOKENIZER_PATH="${WORKING_DIR}/bpe_dialog_new" # vocab path

BATCH_SIZE=16
TRAIN_ITER=-1 # total number of train iterations, if set to -1, the iterations depend on the training epochs (epochs * data_size / (batch_size * grad_acc) )
EPOCHS=3

ENC_LEN=128 # max input length of encoder
DEC_LEN=128 # max input length of decoder


OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --model-parallel-size ${MP_SIZE}"
OPTS+=" --batch-size ${BATCH_SIZE}"
OPTS+=" --epochs ${EPOCHS}"
OPTS+=" --gradient-accumulation-steps ${GRAD_ACC}"
OPTS+=" --enc-seq-length ${ENC_LEN}"
OPTS+=" --dec-seq-length ${DEC_LEN}"
OPTS+=" --train-iters ${TRAIN_ITER}"
OPTS+=" --save ${SAVE_PATH}"
OPTS+=" --log-file ${LOG_FILE}"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --no_load_strict"
OPTS+=" --data-path ${DATA_PATH}"
OPTS+=" --distributed-backend nccl"
OPTS+=" --lr ${LR}"
OPTS+=" --lr-decay-style noam"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --warmup ${WM}"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
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
OPTS+=" --do-eval"
# OPTS+=" --eval-generation" # run the evaluation of generation
OPTS+=" --train-ratio 1"

CMD="torchrun --master_port 1234 --nproc_per_node ${NUM_GPUS_PER_WORKER} ${WORKING_DIR}/src/eva_finetune.py ${OPTS}"

echo ${CMD}
mkdir -p ${SAVE_PATH}
${CMD} 2>&1 | tee ${SAVE_PATH}/train_log
