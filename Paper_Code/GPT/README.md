# GPT Pre-training From Scratch

This folder contains a simple script to pre-train a GPT-2 like model from scratch using the Hugging Face `transformers` library.

## Prerequisites

1. Install the required Python packages:

```bash
pip install -r requirements.txt
```

2. Prepare a training dataset:
   - Create a text file (e.g., `data.txt`) with your training corpus.
   - Each line or block of text will be used for training.

## Usage

Run the `pretrain_gpt.py` script with your training file:

```bash
python pretrain_gpt.py \
    --train_file data.txt \
    --output_dir ./my-gpt-model \
    --num_train_epochs 3 \
    --batch_size 8 \
    --block_size 128 \
    --n_layer 6 \
    --n_head 8 \
    --n_embd 512
```

### Arguments

- `--train_file`: Path to your text file for training (Required).
- `--output_dir`: Directory to save the trained model (Default: `./gpt2-scratch`).
- `--n_layer`: Number of transformer layers (Default: 6).
- `--n_head`: Number of attention heads (Default: 8).
- `--n_embd`: Hidden size/embedding dimension (Default: 512).
- `--block_size`: Maximum sequence length (context window) (Default: 128).
- `--batch_size`: Batch size for training (Default: 8).
- `--learning_rate`: Learning rate (Default: 5e-5).

## Note

- This script uses the `gpt2` tokenizer by default. If you want to train a tokenizer from scratch on your data, you should do that separately and pass the path to `--tokenizer_name`.
- The model architecture is initialized randomly (not pre-trained weights), so it learns entirely from your provided text file.
