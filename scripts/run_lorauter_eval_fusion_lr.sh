#!/bin/bash
# Run lorauter_eval_llama3_intfloat.py on cuda:0 with specified arguments and log output

LOGFILE="lorauter_eval_llama3_intfloat_fusion_lr.log"
nohup env CUDA_VISIBLE_DEVICES=1 python3 -u lorauter_eval_llama3_intfloat_lr.py \
    --data_path dataset/combined_test.json \
    --res_path results_new_emb/lr_uniform.json \
    --config_path config/config_large.json \
    --lora_num 3 \
    --model_size 7b \
    --eval_type mixture \
    > "$LOGFILE" 2>&1 &

echo "Started lorauter_eval_llama3_intfloat_lr.py on cuda:1. Logs: $LOGFILE"
