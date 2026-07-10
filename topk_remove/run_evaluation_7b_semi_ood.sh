#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICE="${1:-${CUDA_DEVICE:-1}}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

echo "Using CUDA device: $CUDA_DEVICE"

python3 lorauter_eval.py \
    --data_path dataset/combined_test.json \
    --res_path results_topk/fusion_7b_semi_ood.json \
    --config_path config/config_large.json \
    --lora_num 3 \
    --batch_size 1 \
    --model_size 7b \
    --eval_type mixture \
    --semi_ood True
