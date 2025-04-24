#!/bin/bash
MASTER_PORT=$(python find_free_port.py)
echo $MASTER_PORT > /tmp/master_port.$SLURM_JOB_ID
CONF_PATH="conf/bioarxiv"
CONF_NAME="dpo_rank1"

for ((round=0;round<=30;round++)); do
    python generate_samples.py --config-path $CONF_PATH --config-name $CONF_NAME round_number=$round
    python client_feedback.py --config-path $CONF_PATH --config-name $CONF_NAME round_number=$round
    accelerate launch --main_process_port $MASTER_PORT --config-file fsdp_config.yml dpo_training.py --config-path $CONF_PATH --config-name $CONF_NAME round_number=$round
    python model_merge.py  --config-path $CONF_PATH --config-name $CONF_NAME round_number=$round
    python generate_samples_eval.py  --config-path $CONF_PATH --config-name $CONF_NAME round_number=$round
    python evaluation.py --config-path $CONF_PATH --config-name $CONF_NAME round_number=$round
done