#!/bin/bash
# Run LoRAuter Arrow evaluation on CUDA device 0, keep running after terminal closes

export CUDA_VISIBLE_DEVICES=0
nohup python3 lorauter_eval_llama3.py \
    --data_path dataset/combined_test.json \
    --res_path results_llama3/arrow.json \
    --config_path config/config_large.json \
    --lora_num 3 \
    --model_size 7b \
    --eval_type arrow \
    > arrow_eval.log 2>&1 &
echo "Started Arrow evaluation in background. Log: arrow_eval.log"
