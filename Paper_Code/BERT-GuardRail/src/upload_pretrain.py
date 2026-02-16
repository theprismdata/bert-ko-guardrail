"""사전학습 BERT(MLM) 모델을 HuggingFace Hub에 업로드."""

import argparse
import json
import math
import os

from transformers import AutoModelForMaskedLM, AutoTokenizer
from huggingface_hub import HfApi


def load_trainer_state(model_dir: str) -> dict | None:
    """trainer_state.json에서 학습 메트릭 추출."""
    # 최신 checkpoint 내부 또는 model_dir 직접 탐색
    for candidate in [model_dir, *sorted(
        [os.path.join(model_dir, d) for d in os.listdir(model_dir)
         if d.startswith("checkpoint-")],
        key=lambda p: int(p.split("-")[-1]),
        reverse=True,
    )]:
        path = os.path.join(candidate, "trainer_state.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
    return None


def extract_mlm_metrics(state: dict) -> dict:
    """trainer_state에서 최종 MLM 메트릭 추출."""
    logs = state.get("log_history", [])

    eval_logs = [l for l in logs if "eval_loss" in l]
    train_logs = [l for l in logs if "loss" in l and "eval_loss" not in l]

    result = {
        "global_step": state.get("global_step", 0),
        "epoch": state.get("epoch", 0),
    }

    if eval_logs:
        last_eval = eval_logs[-1]
        result["eval_loss"] = last_eval["eval_loss"]
        result["eval_perplexity"] = math.exp(min(last_eval["eval_loss"], 20))

    if train_logs:
        last_train = train_logs[-1]
        result["train_loss"] = last_train["loss"]
        result["train_perplexity"] = math.exp(min(last_train["loss"], 20))

    return result


def _fmt(v: float) -> str:
    return f"{v:.4f}"


def create_model_card(repo_id: str, config, mlm_metrics: dict) -> str:
    """사전학습 BERT 모델 카드 생성."""
    repo_name = repo_id.split("/")[-1]

    eval_loss = mlm_metrics.get("eval_loss")
    eval_ppl = mlm_metrics.get("eval_perplexity")
    train_loss = mlm_metrics.get("train_loss")
    train_ppl = mlm_metrics.get("train_perplexity")

    metrics_table = "| Split | Loss | Perplexity |\n|-------|-----:|-----------:|\n"
    if train_loss is not None:
        metrics_table += f"| Train | {_fmt(train_loss)} | {_fmt(train_ppl)} |\n"
    if eval_loss is not None:
        metrics_table += f"| Eval | {_fmt(eval_loss)} | {_fmt(eval_ppl)} |\n"

    yaml_metrics = ""
    if eval_loss is not None:
        yaml_metrics = f"""model-index:
- name: {repo_name}
  results:
  - task:
      type: fill-mask
      name: Masked Language Modeling
    metrics:
    - name: Eval Loss
      type: loss
      value: {_fmt(eval_loss)}
    - name: Eval Perplexity
      type: perplexity
      value: {_fmt(eval_ppl)}"""

    return f"""---
language:
- ko
license: gpl-3.0
tags:
- bert
- masked-language-model
- korean
- pretrained
metrics:
- perplexity
pipeline_tag: fill-mask
{yaml_metrics}
---

# {repo_name}

한국어 텍스트로 사전학습된 BERT (Masked Language Model) 입니다.
가드레일 분류 모델의 백본으로 사용되며, fill-mask 태스크로도 활용 가능합니다.

## 모델 정보

| 항목 | 값 |
|------|-----|
| Architecture | BertForMaskedLM |
| Hidden Size | {config.hidden_size} |
| Layers | {config.num_hidden_layers} |
| Attention Heads | {config.num_attention_heads} |
| Intermediate Size | {config.intermediate_size} |
| Vocab Size | {config.vocab_size:,} |
| Max Length | {config.max_position_embeddings} tokens |
| Total Steps | {mlm_metrics.get('global_step', 'N/A'):,} |
| Epochs | {mlm_metrics.get('epoch', 0):.1f} |

## 사전학습 성능

{metrics_table}

## 학습 코퍼스

| 코퍼스 | 크기 | 설명 |
|--------|------|------|
| injection_corpus.txt | 65MB | 프롬프트 인젝션 데이터 |
| external_all.txt | 9.6MB | KoSBi v2 + K-MHaS + BEEP! |
| all_combined.txt | 15MB | 전체 통합 코퍼스 |

**총 ~90MB** 한국어 텍스트

## 사용 방법

### Fill-Mask

```python
from transformers import pipeline

fill_mask = pipeline("fill-mask", model="{repo_id}")
results = fill_mask("한국의 수도는 [MASK]입니다.")
for r in results:
    print(f"  {{r['token_str']}}: {{r['score']:.4f}}")
```

### 분류 모델 백본으로 사용

```python
from transformers import BertConfig, BertForSequenceClassification

config = BertConfig.from_pretrained("{repo_id}", num_labels=11)
model = BertForSequenceClassification.from_pretrained(
    "{repo_id}", config=config, ignore_mismatched_sizes=True
)
```

## 학습 설정

- **Tokenizer**: WordPiece (vocab_size=32,000)
- **Optimizer**: AdamW
- **Scheduler**: Cosine with warmup
- **MLM Probability**: 15%

## 관련 모델

이 사전학습 모델을 기반으로 파인튜닝된 분류 모델:
- 한국어 가드레일 11-class 분류 모델 (혐오발언 + 프롬프트 인젝션 탐지)

## 라이선스

GPL-3.0 License

## Citation

```bibtex
@misc{{{repo_name},
  author = {{PrismData}},
  title = {{Korean BERT Pretrained (MLM)}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./outputs/pretrain")
    parser.add_argument("--repo_id", type=str, required=True,
                        help="예: username/bert-ko-pretrained")
    parser.add_argument("--token", type=str, required=True)
    args = parser.parse_args()

    print(f"사전학습 BERT 업로드: {args.repo_id}")

    # 모델 로드
    model = AutoModelForMaskedLM.from_pretrained(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"파라미터 수: {param_count:,}")
    print(f"Hidden: {model.config.hidden_size}, Layers: {model.config.num_hidden_layers}")

    # 학습 메트릭 추출
    state = load_trainer_state(args.model_dir)
    if state:
        mlm_metrics = extract_mlm_metrics(state)
        print(f"Steps: {mlm_metrics['global_step']:,}, Epoch: {mlm_metrics['epoch']:.1f}")
        if "eval_loss" in mlm_metrics:
            print(f"Eval  - Loss: {mlm_metrics['eval_loss']:.4f}, PPL: {mlm_metrics['eval_perplexity']:.2f}")
        if "train_loss" in mlm_metrics:
            print(f"Train - Loss: {mlm_metrics['train_loss']:.4f}, PPL: {mlm_metrics['train_perplexity']:.2f}")
    else:
        print("trainer_state.json 미발견, 메트릭 없이 업로드")
        mlm_metrics = {}

    # 모델 카드 생성
    model_card = create_model_card(args.repo_id, model.config, mlm_metrics)

    # 업로드
    print("모델 업로드 중...")
    api = HfApi(token=args.token)
    model.push_to_hub(args.repo_id, token=args.token, private=False)
    tokenizer.push_to_hub(args.repo_id, token=args.token, private=False)

    print("모델 카드 업로드 중...")
    api.upload_file(
        path_or_fileobj=model_card.encode(),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
    )

    print(f"완료: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
