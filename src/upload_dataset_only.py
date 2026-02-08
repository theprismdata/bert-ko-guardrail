"""Hugging Face 데이터셋 업로드 전용 스크립트."""

import argparse
from datasets import load_dataset
from huggingface_hub import HfApi

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/classification", help="데이터 경로")
    parser.add_argument("--repo_id", help="HuggingFace 저장소 ID (예: username/dataset-name)")
    parser.add_argument("--token", required=True, help="HF Write Token")
    args = parser.parse_args()

    # 사용자명 자동 감지 (repo_id가 없을 때)
    if not args.repo_id:
        try:
            api = HfApi(token=args.token)
            user_info = api.whoami()
            username = user_info['name']
            print(f"Detected username: {username}")
            args.repo_id = f"{username}/hate-speech-ko-style"
        except Exception as e:
            print(f"Failed to detect username: {e}")
            return

    print(f"Loading data from {args.data_dir}...")
    dataset = load_dataset("csv", data_files={
        "train": f"{args.data_dir}/train.csv",
        "validation": f"{args.data_dir}/validation.csv",
    })

    print(f"Uploading to {args.repo_id}...")
    try:
        dataset.push_to_hub(args.repo_id, token=args.token)
        print(f"Successfully uploaded to https://huggingface.co/datasets/{args.repo_id}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
