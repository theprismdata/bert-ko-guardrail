#!/bin/bash
# 3개 데이터셋 통합 → 10-class 분류 데이터셋 생성
# 사전 조건: run_prepare_data.sh, run_prepare_external.sh 실행 완료
# 사용법: bash scripts/run_prepare_multiclass.sh

set -e

export TOKENIZERS_PARALLELISM=false

python src/prepare_multiclass.py --data_dir ./data
