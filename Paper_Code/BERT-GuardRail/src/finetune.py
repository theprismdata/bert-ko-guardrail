"""사전학습 BERT를 이용한 문장 분류 파인튜닝 스크립트."""

import argparse
import json
import math
import os

import evaluate
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
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
    precision_metric = evaluate.load("precision")
    recall_metric = evaluate.load("recall")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)["accuracy"]
        precision_macro = precision_metric.compute(
            predictions=preds, references=labels, average="macro"
        )["precision"]
        precision_weighted = precision_metric.compute(
            predictions=preds, references=labels, average="weighted"
        )["precision"]
        recall_macro = recall_metric.compute(
            predictions=preds, references=labels, average="macro"
        )["recall"]
        recall_weighted = recall_metric.compute(
            predictions=preds, references=labels, average="weighted"
        )["recall"]
        f1_macro = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        f1_weighted = f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
        return {
            "accuracy": acc,
            "precision_macro": precision_macro,
            "precision_weighted": precision_weighted,
            "recall_macro": recall_macro,
            "recall_weighted": recall_weighted,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            # 기존 설정(metric_for_best_model=f1 등)과의 호환을 위한 별칭
            "f1": f1_weighted,
        }

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

    # 테스트셋/검증셋 평가 + HF 업로드용 메트릭 파일 저장
    metrics_payload = {
        "num_labels": num_labels,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
    }

    def _save_confusion_matrix(labels, preds, target_names, split_name):
        """Confusion matrix를 정규화 heatmap으로 저장."""
        cm = confusion_matrix(labels, preds)
        # 행(실제) 기준 정규화 (recall 관점)
        cm_norm = cm.astype(float)
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm_norm / row_sums

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names,
            ax=ax, vmin=0, vmax=1,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion Matrix ({split_name}) - Row Normalized")
        plt.tight_layout()
        path = os.path.join(training_args.output_dir, f"confusion_matrix_{split_name}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"Saved confusion matrix to {path}")

    def _eval_with_per_class(split_name, dataset):
        """평가 + per-class classification_report + confusion matrix 생성."""
        agg = trainer.evaluate(dataset, metric_key_prefix=split_name)
        # per-class 메트릭
        preds_output = trainer.predict(dataset)
        preds = np.argmax(preds_output.predictions, axis=-1)
        labels = preds_output.label_ids
        target_names = [id2label[i] for i in range(num_labels)]
        report = classification_report(
            labels, preds, target_names=target_names, output_dict=True, zero_division=0
        )
        agg["per_class"] = {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in target_names
            if name in report
        }
        # confusion matrix 저장
        _save_confusion_matrix(labels, preds, target_names, split_name)
        return agg

    if "validation" in datasets:
        metrics_payload["validation"] = _eval_with_per_class("validation", datasets["validation"])
    if "test" in datasets:
        results = _eval_with_per_class("test", datasets["test"])
        print(f"Test results: {results}")
        metrics_payload["test"] = results

    metrics_path = os.path.join(training_args.output_dir, "classification_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
