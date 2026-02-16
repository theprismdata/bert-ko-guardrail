#!/bin/bash
# BERT MLM 사전학습 (랜덤 초기화)
# 입력: outputs/tokenizer + data/corpus → 출력: outputs/pretrain/
# 사용법: bash scripts/run_pretrain.sh

set -e

TOKENIZERS_PARALLELISM=false \
WANDB_DISABLED=true \
python3 src/pretrain.py \
    --model_config ./configs/model_config.json \
    --train_config ./configs/pretrain_config.json \
    --tokenizer_dir ./outputs/tokenizer \
    --data_dir ./data/corpus
