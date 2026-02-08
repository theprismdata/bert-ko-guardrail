# CLAUDE.md

## Project Overview

한국어 혐오발언 탐지를 위한 BERT 사전학습 및 10-class 분류 프로젝트.
HuggingFace Transformers + Trainer API 기반, 단일 GPU(MPS/CUDA) 환경.

## Pipeline

```bash
# 0. 데이터 준비 (3개 데이터셋 자동 다운로드 + 10-class 통합)
bash scripts/run_prepare_multiclass.sh

# 1. 토크나이저 학습 (WordPiece, vocab=32K)
bash scripts/run_tokenizer.sh

# 2. MLM 사전학습
bash scripts/run_pretrain.sh

# 3. 분류 파인튜닝
bash scripts/run_finetune.sh
```

## 10-Class Labels

| # | Label | 설명 |
|---|-------|------|
| 0 | SAFE | 정상 발화 |
| 1 | ORIGIN | 출신 지역 차별 |
| 2 | PHYSICAL | 외모/신체/장애 차별 |
| 3 | POLITICS | 정치적 편향 |
| 4 | PROFANITY | 욕설/비속어 |
| 5 | AGE | 나이/세대 차별 |
| 6 | GENDER | 성별/성적지향 차별 |
| 7 | RACE | 인종/민족 차별 |
| 8 | RELIGION | 종교 차별 |
| 9 | SOCIAL | 사회적 지위/학력/가족 차별 |

## Notes

- Apple Silicon (MPS): `fp16: false`, `dataloader_num_workers: 0`
- CUDA GPU: `fp16: true`, `dataloader_num_workers: 4`
