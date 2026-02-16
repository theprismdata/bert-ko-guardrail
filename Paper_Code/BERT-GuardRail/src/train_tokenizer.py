"""WordPiece 토크나이저 학습 스크립트."""

import argparse
import os
from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers


def main():
    parser = argparse.ArgumentParser(description="Train a WordPiece tokenizer")
    parser.add_argument("--corpus_dir", type=str, required=True,
                        help="Directory containing .txt files for training")
    parser.add_argument("--output_dir", type=str, default="./outputs/tokenizer",
                        help="Directory to save the trained tokenizer")
    parser.add_argument("--vocab_size", type=int, default=32000,
                        help="Vocabulary size")
    parser.add_argument("--min_frequency", type=int, default=2,
                        help="Minimum token frequency to be included in vocab")
    args = parser.parse_args()

    # 코퍼스 파일 수집
    corpus_files = sorted(str(p) for p in Path(args.corpus_dir).glob("*.txt"))
    if not corpus_files:
        raise FileNotFoundError(f"No .txt files found in {args.corpus_dir}")
    print(f"Found {len(corpus_files)} corpus files")

    # WordPiece 토크나이저 구성
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))

    tokenizer.normalizer = normalizers.Sequence([
        normalizers.NFD(),
        normalizers.StripAccents(),
        normalizers.Lowercase(),
    ])
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    trainer = trainers.WordPieceTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens,
        continuing_subword_prefix="##",
    )

    # 학습
    print(f"Training tokenizer (vocab_size={args.vocab_size})...")
    tokenizer.train(corpus_files, trainer=trainer)
    print(f"Trained vocab size: {tokenizer.get_vocab_size()}")

    # 저장
    os.makedirs(args.output_dir, exist_ok=True)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_path)

    # vocab.txt 추출 (BertTokenizerFast 호환)
    vocab = tokenizer.get_vocab()
    vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])
    vocab_path = os.path.join(args.output_dir, "vocab.txt")
    with open(vocab_path, "w", encoding="utf-8") as f:
        for token, _ in vocab_sorted:
            f.write(token + "\n")

    # HuggingFace AutoTokenizer 호환 저장
    from transformers import BertTokenizerFast
    hf_tokenizer = BertTokenizerFast(
        vocab_file=vocab_path,
        tokenizer_file=tokenizer_path,
        do_lower_case=True,
    )
    hf_tokenizer.save_pretrained(args.output_dir)

    print(f"Tokenizer saved to {args.output_dir}")
    print(f"Vocab size: {hf_tokenizer.vocab_size}")


if __name__ == "__main__":
    main()
