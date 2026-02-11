"""BERT MLM 사전학습 스크립트.

BertConfig로 모델을 랜덤 초기화하고, Masked Language Modeling으로 사전학습한다.
"""

import argparse
import json

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import (
    AutoTokenizer,
    BertConfig,
    # BertForMaskedLM,  # 대체됨
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from bert_model import CustomBertForMaskedLM

from data_utils import load_text_files


def tokenize_function(examples, tokenizer, max_seq_length):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_seq_length,
        padding=False,
        return_special_tokens_mask=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Pre-train BERT with MLM")
    parser.add_argument("--model_config", type=str, default="./configs/model_config.json",
                        help="Path to model architecture config JSON")
    parser.add_argument("--train_config", type=str, default="./configs/pretrain_config.json",
                        help="Path to training hyperparameter config JSON")
    parser.add_argument("--tokenizer_dir", type=str, required=True,
                        help="Path to trained tokenizer directory")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing .txt corpus files")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory from config")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    # 설정 로드
    with open(args.model_config) as f:
        model_cfg = json.load(f)
    with open(args.train_config) as f:
        train_cfg = json.load(f)

    mlm_probability = train_cfg.pop("mlm_probability", 0.15)
    max_seq_length = train_cfg.pop("max_seq_length", 256)
    
    # transformers 버전에 따라 save_safetensors 인자를 지원하지 않을 수 있음
    train_cfg.pop("save_safetensors", None)

    if args.output_dir:
        train_cfg["output_dir"] = args.output_dir

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    # vocab_size를 토크나이저에 맞게 조정
    model_cfg["vocab_size"] = tokenizer.vocab_size

    # 모델 생성 (랜덤 초기화) - Custom Model 사용
    config = BertConfig(**model_cfg)
    model = CustomBertForMaskedLM(config)
    # model = BertForMaskedLM(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model initialized: {param_count:,} parameters")

    # 데이터 로드 및 토크나이징
    raw_datasets = load_text_files(args.data_dir)
    tokenized_datasets = {}
    for split, ds in raw_datasets.items():
        tokenized_datasets[split] = ds.map(
            lambda examples: tokenize_function(examples, tokenizer, max_seq_length),
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing {split}",
        )

    # Data collator (동적 마스킹)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_probability,
    )

    # TrainingArguments
    training_args = TrainingArguments(**train_cfg)
    # CustomBertForMaskedLM은 embedding·MLM head weight tying 사용 → safetensors 저장 시 공유 메모리 오류.
    # 체크포인트/최종 저장을 PyTorch .bin으로 하기 위해 safe_serialization 비활성화.
    setattr(training_args, "safe_serialization", False)

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        processing_class=tokenizer,
    )

    # 학습
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # 최종 모델 저장
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()
