"""사전학습된 BERT MLM 모델 테스트 (fill-mask)."""

import argparse
import json
import os

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig, BertForMaskedLM


def load_mlm_model(model_dir, tokenizer_dir, model_config_path=None):
    tokenizer_path = tokenizer_dir or model_dir
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    config_path = os.path.join(model_dir, "config.json")
    if os.path.isfile(config_path):
        model = AutoModelForMaskedLM.from_pretrained(model_dir)
        return tokenizer, model
    if model_config_path and os.path.isfile(model_config_path):
        with open(model_config_path) as f:
            cfg = json.load(f)
        cfg["vocab_size"] = tokenizer.vocab_size
        config = BertConfig(**cfg)
        model = BertForMaskedLM(config)
        weights_path = os.path.join(model_dir, "model.safetensors")
        if os.path.isfile(weights_path):
            from safetensors.torch import load_file
            state = load_file(weights_path)
            model.load_state_dict(state, strict=False)
        else:
            raise FileNotFoundError(f"No config.json or model.safetensors in {model_dir}")
        return tokenizer, model
    raise FileNotFoundError(
        f"No config.json in {model_dir}. Use --model_config or save model with pretrain.py."
    )


def main():
    parser = argparse.ArgumentParser(description="Test pretrained BERT MLM (fill-mask)")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Path to pretrained model dir (outputs/pretrain)")
    parser.add_argument("--tokenizer_dir", type=str, default="./outputs/tokenizer",
                        help="Path to tokenizer (default: ./outputs/tokenizer)")
    parser.add_argument("--model_config", type=str, default="./configs/model_config.json",
                        help="Model config JSON when model_dir has no config.json")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top predictions to show per [MASK]")
    args = parser.parse_args()

    tokenizer, model = load_mlm_model(
        args.model_dir, args.tokenizer_dir, args.model_config
    )
    model.eval()

    # 테스트 문장 (한국어, [MASK] 위치에서 예측)
    examples = [
        "오늘 날씨가 정말 [MASK].",
        "그 사람은 [MASK] 일을 잘한다.",
        "한국어 [MASK] 모델을 학습시켰다.",
    ]

    for text in examples:
        print(f"\n입력: {text}")
        inputs = tokenizer(text, return_tensors="pt")
        mask_token_id = tokenizer.mask_token_id
        mask_idx = (inputs["input_ids"][0] == mask_token_id).nonzero(as_tuple=True)[0].item()

        with torch.no_grad():
            logits = model(**inputs).logits[0, mask_idx]
        probs = logits.softmax(dim=-1)
        top_k = probs.topk(args.top_k)

        for rank, (idx, p) in enumerate(zip(top_k.indices.tolist(), top_k.values.tolist()), 1):
            token = tokenizer.decode([idx])
            print(f"  {rank}. {token!r} (p={p:.4f})")
    print()


if __name__ == "__main__":
    main()
