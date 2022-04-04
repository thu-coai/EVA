#! /bin/bash
WORKING_DIR=/home/leichangsong/eva-huggingface
TEMP=0.9
#If TOPK/TOPP are 0 it defaults to greedy sampling, top-k will also override top-p
TOPK=0
TOPP=0.9
NUM_BEAMS=4
CKPT_PATH="/home/leichangsong/another.pt"
RULE_PATH="${WORKING_DIR}/rules"
TOKENIZER_PATH="${WORKING_DIR}/vocab"
CONFIG_PATH="${WORKING_DIR}/model_config/eva2.0_model_config.json"


OPTS=""
OPTS+=" --model-config ${CONFIG_PATH}"
OPTS+=" --no_load_strict"
OPTS+=" --weight-decay 1e-2"
OPTS+=" --clip-grad 1.0"
OPTS+=" --load ${CKPT_PATH}"
OPTS+=" --temperature ${TEMP}"
OPTS+=" --top_k ${TOPK}"
OPTS+=" --top_p ${TOPP}"
OPTS+=" --num-beams ${NUM_BEAMS}"
OPTS+=" --rule-path ${RULE_PATH}"
OPTS+=" --tokenizer-path ${TOKENIZER_PATH}"
OPTS+=" --length-penalty 1.6"
OPTS+=" --repetition-penalty 1.6"

CMD="python ../eva_interactive.py ${OPTS}"

echo ${CMD}
${CMD}
