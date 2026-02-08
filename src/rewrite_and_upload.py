"""데이터셋 말투 변환(Rule-based) 및 HuggingFace 업로드 스크립트."""

import argparse
import os
import re
import random
from datasets import load_dataset

# 간단한 말투 변환 규칙
def apply_style_transfer(text):
    original = text
    
    # 1. 마침표 제거
    text = text.rstrip(".")

    # 2. 종결어미 변환 (음슴체/반말/구어체)
    # 안전한 변환을 위해 자주 쓰는 패턴 위주로 교체 (Regex)
    patterns = [
        (r"(하|해)요$", r"\1"),        # 해요 -> 해
        (r"(에|예)요$", "야"),         # 에요 -> 야
        (r"죠$", "지"),               # 죠 -> 지
        (r"구요$", "구"),             # 구요 -> 구
        (r"까요$", "까"),             # 까요 -> 까
        (r"나요$", "나"),             # 나요 -> 나
        (r"가요$", "가"),             # 가요 -> 가
        (r"와요$", "와"),             # 와요 -> 와
        (r"봐요$", "봐"),             # 봐요 -> 봐
        (r"줘요$", "줘"),             # 줘요 -> 줘
        (r"세요$", "라"),             # 세요 -> 라 (명령형)
        (r"(겠|었|였)?(습니다|니다)$", r"\1음"),  # 습니다/니다 -> 음
        (r"(입|입니)다$", "임"),       # 입니다 -> 임
        (r"입니까$", "임?"),          # 입니까 -> 임?
        (r"습니까$", "음?"),          # 습니까 -> 음?
        (r"요$", ""),                 # 그 외 요 제거 (주의)
    ]
    
    for pat, repl in patterns:
        text = re.sub(pat, repl, text)
    
    # 3. 요즘 강조 부사로 교체 (랜덤 적용)
    # "정말/매우/진짜/너무" -> "존나/개/핵"
    adverbs = ["정말", "매우", "진짜", "너무", "참"]
    new_adverbs = ["존나", "개", "핵", "씹"]
    
    for adv in adverbs:
        if adv in text and random.random() < 0.5:  # 50% 확률로 교체
            text = text.replace(adv, random.choice(new_adverbs))

    # 4. ㅋㅋㅋ/ㅎㅎㅎ 추가 (랜덤)
    if random.random() < 0.2:
        text += random.choice([" ㅋㅋ", " ㅋㅋㅋ", " ㅎ", " ㄷㄷ"])

    return text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/classification", help="원본 데이터 경로")
    parser.add_argument("--output_dir", default="data/classification_style", help="변환된 데이터 저장 경로")
    parser.add_argument("--repo_id", help="HuggingFace 저장소 ID (지정하면 업로드)")
    parser.add_argument("--token", help="HF Write Token (로그인 안 된 경우)")
    args = parser.parse_args()

    # 데이터 로드
    print(f"Loading data from {args.data_dir}...")
    dataset = load_dataset("csv", data_files={
        "train": f"{args.data_dir}/train.csv",
        "validation": f"{args.data_dir}/validation.csv",
    })

    # 변환 적용
    print("Applying style transfer...")
    def transform(batch):
        return {"text": [apply_style_transfer(t) for t in batch["text"]]}
    
    dataset = dataset.map(transform, batched=True)

    # 변환 결과 일부 확인
    print("-" * 50)
    print("Preview (First 10 examples):")
    for i in range(10):
        print(f"Styled  : {dataset['train'][i]['text']}")
    print("-" * 50)

    # 로컬 저장
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"Saving to {args.output_dir}...")
        for split in dataset:
            output_path = os.path.join(args.output_dir, f"{split}.csv")
            dataset[split].to_csv(output_path, index=False)
        print("Saved!")

    # 업로드 (선택)
    if args.repo_id:
        print(f"Uploading to {args.repo_id}...")
        try:
            dataset.push_to_hub(args.repo_id, token=args.token)
            print("Successfully uploaded!")
        except Exception as e:
            print(f"Upload failed: {e}")
            print("Try 'huggingface-cli login' first or provide --token")
    else:
        print("Skipping upload (provide --repo_id to upload)")

if __name__ == "__main__":
    main()
