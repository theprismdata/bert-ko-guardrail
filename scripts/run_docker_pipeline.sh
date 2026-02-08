#!/bin/bash
set -e

# 1. Tokenizer Training
echo "=== 1. Tokenizer Training ==="
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c "pip install transformers datasets scikit-learn && bash scripts/run_tokenizer.sh"

# 2. Pretraining
echo "=== 2. Pretraining ==="
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c "pip install transformers datasets scikit-learn accelerate && bash scripts/run_pretrain.sh"

# 3. Finetuning
echo "=== 3. Finetuning ==="
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c "pip install transformers datasets scikit-learn accelerate && bash scripts/run_finetune.sh"

echo "=== ALL DONE ==="
