#! /bin/bash
WORKING_DIR=/home/guyuxian/EVA

TEMP=0.9

TOPK=0
TOPP=0.9
NUM_BEAMS=4
CKPT_PATH="${WORKING_DIR}/checkpoints/eva2.0-hf"


OPTS=""
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --do-sample"
OPTS+=" --temperature ${TEMP}"
OPTS+=" --top_k ${TOPK}"
OPTS+=" --top_p ${TOPP}"
OPTS+=" --num-beams ${NUM_BEAMS}"
OPTS+=" --length-penalty 1.6"
OPTS+=" --repetition-penalty 1.6"


CMD="python ${WORKING_DIR}/src/eva_interactive.py ${OPTS}"

echo ${CMD}
${CMD}
