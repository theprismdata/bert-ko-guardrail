#!/bin/bash
# 문장 분류 파인튜닝 (INJECTION / LEGIT)
# 입력: outputs/pretrain + data/classification/ → 출력: outputs/finetune/
# 사용법: bash scripts/run_finetune.sh

set -e

TOKENIZERS_PARALLELISM=false \
WANDB_DISABLED=true \
python3 src/finetune.py \
    --model_dir ./outputs/pretrain \
    --train_config ./configs/finetune_config.json \
    --data_dir ./data/classification_v3 \
    --text_col text \
    --label_col label
