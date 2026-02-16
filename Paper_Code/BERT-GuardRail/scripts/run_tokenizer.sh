#!/bin/bash
# WordPiece 토크나이저 학습
# 입력: data/corpus/*.txt → 출력: outputs/tokenizer/
# 사용법: bash scripts/run_tokenizer.sh

set -e

TOKENIZERS_PARALLELISM=false \
python3 src/train_tokenizer.py \
    --corpus_dir ./data/corpus \
    --output_dir ./outputs/tokenizer \
    --vocab_size 32000 \
    --min_frequency 2
