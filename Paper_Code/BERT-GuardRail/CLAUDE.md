# CLAUDE.md

## Project Overview

**11-Class 가드레일 모델** - 한국어 혐오발언 + 프롬프트 인젝션 통합 탐지 시스템.
HuggingFace Transformers + Trainer API 기반, 단일 GPU(MPS/CUDA) 환경.

## Pipeline

### Option A: 사전학습된 BERT 사용 (빠른 시작)

```bash
# 1. 데이터 준비 (선택, 이미 완료: data/classification_v3)
bash scripts/run_prepare_multiclass.sh

# 2. 가드레일 모델 파인튜닝 (기존 outputs/pretrain 사용)
bash scripts/run_finetune.sh

# 3. 모델 테스트
python3 src/test_classification.py --model_dir ./outputs/finetune
```

### Option B: 처음부터 학습 (Full Pipeline)

```bash
# 1. 데이터 준비
bash scripts/run_prepare_multiclass.sh

# 2. 토크나이저 학습 (WordPiece, vocab=32K)
bash scripts/run_tokenizer.sh

# 3. BERT MLM 사전학습
bash scripts/run_pretrain.sh

# 4. 가드레일 모델 파인튜닝
bash scripts/run_finetune.sh

# 5. 모델 테스트
python3 src/test_classification.py --model_dir ./outputs/finetune
```

## 11-Class Labels

| # | Label | 설명 | Train 샘플 수 |
|---|-------|------|-------------|
| 0 | SAFE | 정상 발화 | 73,587 |
| 1 | ORIGIN | 출신 지역 차별 | 9,924 |
| 2 | PHYSICAL | 외모/신체/장애 차별 | 8,262 |
| 3 | POLITICS | 정치적 편향 | 8,781 |
| 4 | PROFANITY | 욕설/비속어 | 10,708 |
| 5 | AGE | 나이/세대 차별 | 8,934 |
| 6 | GENDER | 성별/성적지향 차별 | 6,262 |
| 7 | RACE | 인종/민족 차별 | 3,161 |
| 8 | RELIGION | 종교 차별 | 3,466 |
| 9 | SOCIAL | 사회적 지위/학력/가족 차별 | 7,392 |
| 10 | INJECTION | 프롬프트 인젝션 | 61,836 |

**총 202,313개** 샘플 (data/classification_v3)

## Data Sources

### 분류 데이터
- **혐오발언 (10-class)**: KoSBi v2, K-MHaS, BEEP! 통합
- **프롬프트 인젝션**: Gemini API로 한글 번역된 영문 데이터셋
- **통합 데이터**: `data/classification_v3/` (train/validation/test)

### MLM 사전학습 코퍼스
- `data/corpus/injection_corpus.txt` (65MB) - 프롬프트 인젝션 데이터
- `data/corpus/external_all.txt` (9.6MB) - KoSBi v2 + K-MHaS + BEEP!
- `data/corpus/all_combined.txt` (15MB) - 전체 통합 코퍼스
- **총 ~90MB** 텍스트 데이터

## Model Configuration

- **Base Model**: 사전학습된 한국어 BERT (`outputs/pretrain`)
- **Architecture**: BertForSequenceClassification (11 classes)
- **Hardware**: 
  - Apple Silicon (MPS): `fp16: false`, `dataloader_num_workers: 0`
  - CUDA GPU: `fp16: true`, `dataloader_num_workers: 4`

## Use Case

이 모델은 LLM 가드레일로 사용되어 다음을 동시에 탐지합니다:
- 사용자 입력의 프롬프트 인젝션 시도
- 모델 출력의 혐오발언/유해 컨텐츠 (10개 카테고리)