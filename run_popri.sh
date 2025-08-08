#!/bin/bash
CONF_PATH="conf/bioarxiv"
CONF_NAME="dpo_eps1"

for ((round=0;round<=30;round++)); do
    python generate_samples.py --config-path $CONF_PATH --config-name $CONF_NAME round_number=$round
    python client_feedback.py --config-path $CONF_PATH --config-name $CONF_NAME round_number=$round
    python dpo_training.py --config-path $CONF_PATH --config-name $CONF_NAME round_number=$round
    python model_merge.py  --config-path $CONF_PATH --config-name $CONF_NAME round_number=$round
    python generate_samples_eval.py  --config-path $CONF_PATH --config-name $CONF_NAME round_number=$round
    python evaluation.py --config-path $CONF_PATH --config-name $CONF_NAME round_number=$round
done
