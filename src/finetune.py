"""사전학습 BERT를 이용한 문장 분류 파인튜닝 스크립트."""

import argparse
import json
import math

import evaluate
import numpy as np
from transformers import (
    AutoTokenizer,
    BertConfig,
    BertForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from data_utils import build_label_map, load_classification_data


def main():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for sentence classification")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to pre-trained BERT checkpoint")
    parser.add_argument("--train_config", type=str, default="./configs/finetune_config.json",
                        help="Path to fine-tuning hyperparameter config JSON")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing train.csv/tsv, validation.csv/tsv")
    parser.add_argument("--text_col", type=str, default="text",
                        help="Name of the text column in CSV/TSV")
    parser.add_argument("--label_col", type=str, default="label",
                        help="Name of the label column in CSV/TSV")
    parser.add_argument("--num_labels", type=int, default=None,
                        help="Number of classes (auto-detected if not set)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory from config")
    args = parser.parse_args()

    # 설정 로드
    with open(args.train_config) as f:
        train_cfg = json.load(f)

    max_seq_length = train_cfg.pop("max_seq_length", 256)
    early_stopping_patience = train_cfg.pop("early_stopping_patience", 2)

    if args.output_dir:
        train_cfg["output_dir"] = args.output_dir

    # 데이터 로드
    datasets = load_classification_data(
        args.data_dir, text_col=args.text_col, label_col=args.label_col
    )

    # 레이블 매핑
    label_map = build_label_map(datasets["train"])
    num_labels = args.num_labels or len(label_map)
    id2label = {v: k for k, v in label_map.items()}
    label2id = label_map
    print(f"Labels ({num_labels}): {label_map}")

    # 문자열 레이블 → 정수 변환
    for split in datasets:
        datasets[split] = datasets[split].map(
            lambda ex: {"label": label_map[ex["label"]]},
            desc=f"Encoding labels ({split})",
        )

    # 토크나이저 & 모델 로드
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    config = BertConfig.from_pretrained(
        args.model_dir,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )
    model = BertForSequenceClassification.from_pretrained(
        args.model_dir, config=config, ignore_mismatched_sizes=True
    )

    # 토크나이징 (CSV에서 NaN/None이 오면 토크나이저 타입 오류 발생하므로 str로 정제)
    def tokenize_fn(examples):
        texts = examples["text"]
        cleaned = []
        for x in texts:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                cleaned.append("")
            else:
                cleaned.append(str(x))
        return tokenizer(
            cleaned,
            truncation=True,
            max_length=max_seq_length,
            padding=False,
        )

    for split in datasets:
        datasets[split] = datasets[split].map(
            tokenize_fn, batched=True, remove_columns=["text"],
            desc=f"Tokenizing {split}",
        )

    # 메트릭
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)
        f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
        return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

    # TrainingArguments
    training_args = TrainingArguments(**train_cfg)

    # Trainer (EarlyStopping: validation F1이 N epoch 연속 개선 없으면 종료)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("validation"),
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
    )

    # 학습
    trainer.train()

    # 저장
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"Fine-tuned model saved to {training_args.output_dir}")

    # 테스트셋 평가
    if "test" in datasets:
        results = trainer.evaluate(datasets["test"], metric_key_prefix="test")
        print(f"Test results: {results}")


if __name__ == "__main__":
    main()
