import argparse
import os
import torch
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from datasets import load_dataset

def train(args):
    # 1. 토크나이저 설정
    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(args.tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. GPT-2 모델 설정
    print("Initializing new GPT-2 model from scratch...")
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        activation_function="gelu_new",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
    )
    
    model = GPT2LMHeadModel(config)
    print(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")

    # 3. 데이터셋 로드 (datasets 라이브러리 사용)
    print(f"Loading dataset from {args.train_file}")
    dataset = load_dataset("text", data_files={"train": args.train_file})

    # 4. 데이터 전처리 (토크나이징 및 블록화)
    print("Tokenizing dataset...")
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=False) # 긴 문장도 자르지 않고 그대로 유지 (뒤에서 자름)

    tokenized_datasets = dataset.map(
        tokenize_function, 
        batched=True, 
        num_proc=1, # Mac 안정성을 위해 1로 설정
        remove_columns=["text"]
    )

    block_size = args.block_size
    print(f"Grouping texts into chunks of {block_size}...")
    
    def group_texts(examples):
        # 모든 텍스트를 하나로 연결
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        
        # block_size 단위로 자르기 (나머지는 버림)
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
            
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=1, # Mac 안정성을 위해 1로 설정
    )
    train_dataset = lm_datasets["train"]

    # 5. Data Collator 설정
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # 6. 학습 설정
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        do_train=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=100,
        prediction_loss_only=True,
        remove_unused_columns=False, # datasets 사용 시 필요할 수 있음
    )

    # 7. Trainer 초기화 및 학습 시작
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print("Starting training...")
    trainer.train()
    
    print(f"Saving model to {args.output_dir}")
    
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train a GPT-2 model from scratch using Hugging Face Transformers")
    
    # 필수 인자
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training text file (.txt)")
    parser.add_argument("--output_dir", type=str, default="./gpt2-scratch", help="Directory to save the model")
    
    # 모델 구조 관련 인자 (기본값: GPT-2 Small 비슷하게)
    parser.add_argument("--n_layer", type=int, default=6, help="Number of transformer layers")
    parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_embd", type=int, default=512, help="Embedding dimension (hidden size)")
    parser.add_argument("--block_size", type=int, default=128, help="Sequence length (context window)")
    
    # 학습 관련 인자
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per device")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X steps")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Limit total amount of checkpoints")
    parser.add_argument("--tokenizer_name", type=str, default="gpt2", help="Pretrained tokenizer name to use")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached training sets")

    args = parser.parse_args()
    
    if not os.path.exists(args.train_file):
        print(f"Error: Training file '{args.train_file}' not found.")
        exit(1)
        
    train(args)
