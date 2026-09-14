# configs/

`src/pretrain.py` / `src/finetune.py` 가 참조하는 설정 파일들.
원래 커밋되지 않아 유실되었던 것을, 배포된 모델
[`prismdata/bert-ko-pretrained`](https://huggingface.co/prismdata/bert-ko-pretrained)
의 `config.json` 및 모델 카드로부터 복원한 것이다.

## 값의 출처 (확정 vs 복원)

### `model_config.json` — 전부 확정 ✅
HF 모델의 `config.json` 과 1:1 일치. 이 설정으로 만든
`BertForMaskedLM` 은 **11,515,904 파라미터**로 배포본과 동일하다.

| 항목 | 값 |
|------|-----|
| hidden_size | 256 |
| num_hidden_layers | 4 |
| num_attention_heads | 4 |
| intermediate_size | 1024 |
| max_position_embeddings | 256 |
| vocab_size | 32000 |

> `pretrain.py` 가 `vocab_size` 를 토크나이저 값으로 덮어쓰므로,
> 토크나이저를 `--vocab_size 32000` 으로 학습하면 자동으로 일치한다.

### `pretrain_config.json` — 일부 확정, 일부 복원 ⚠️
`TrainingArguments(**train_cfg)` 로 그대로 전달된다
(`mlm_probability`, `max_seq_length` 두 키는 스크립트가 먼저 pop 함).

| 항목 | 값 | 출처 |
|------|-----|------|
| max_steps | 50000 | ✅ 모델 카드 (Total Steps) |
| mlm_probability | 0.15 | ✅ 모델 카드 |
| max_seq_length | 256 | ✅ 모델 카드 / config |
| lr_scheduler_type | cosine | ✅ 모델 카드 (Cosine with warmup) |
| optim | adamw_torch | ✅ 모델 카드 (AdamW) |
| output_dir | ./outputs/pretrain | ✅ run_pretrain.sh |
| learning_rate | 5e-4 | ⚠️ **복원 (미기록)** — 소형 BERT 통상값 |
| per_device_train_batch_size | 64 | ⚠️ **복원 (미기록)** |
| warmup_ratio | 0.1 | ⚠️ **복원 (미기록)** |
| weight_decay | 0.01 | ⚠️ **복원 (미기록)** |
| eval_steps / save_steps | 1000 | ⚠️ **복원 (미기록)** |
| seed | 42 | ⚠️ **복원 (미기록)** |

### `finetune_config.json` — 전부 복원 ⚠️
파인튜닝 하이퍼파라미터는 어디에도 기록이 남아있지 않아
**전부 합리적 기본값으로 복원**했다. `finetune.py` 가 실제로
요구하는 키(`max_seq_length`, `early_stopping_patience` + 표준
`TrainingArguments`)는 모두 채워 바로 실행 가능하다.
필요 시 실제 실험값으로 조정할 것.

## 실행

```bash
bash scripts/run_tokenizer.sh   # → outputs/tokenizer (vocab 32000)
bash scripts/run_pretrain.sh    # → outputs/pretrain  (bert-ko-pretrained)
bash scripts/run_finetune.sh    # → outputs/finetune  (INJECTION/LEGIT)
```
