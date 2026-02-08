"""프롬프트 인젝션 데이터셋(geekyrakshit/prompt-injection-dataset)을
기존 다중 분류 데이터(data/classification_mod)에 'INJECTION' 레이블로 추가합니다.
"""

import os
import argparse
import pandas as pd
from datasets import load_dataset


def add_injection_data(output_dir):
    print("Loading geekyrakshit/prompt-injection-dataset...")
    ds = load_dataset("geekyrakshit/prompt-injection-dataset")
    
    # 1. Train 데이터 병합
    train_path = os.path.join(output_dir, "train.csv")
    if os.path.exists(train_path):
        print(f"Reading existing {train_path}...")
        df_existing = pd.read_csv(train_path)
    else:
        print(f"Creating new {train_path}...")
        df_existing = pd.DataFrame(columns=["text", "label"])

    print("Processing injection train data...")
    # 'instruction' 컬럼이 공격 구문으로 추정됨
    # 원본 데이터 구조: {'instruction': ..., 'response': ...}
    # 여기서는 instruction을 text로, label을 'INJECTION'으로 설정
    new_rows = []
    print(ds["train"])
    for row in ds["train"]:
        text = row["prompt"].strip()
        label = row["label"]
        if text:
            if label == 0: #정상
                new_rows.append({"text": text, "label": "NORMAL"})
            else:
                new_rows.append({"text": text, "label": "INJECTION"})
    
    df_new = pd.DataFrame(new_rows)
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    
    # 중복 제거 (text 기준)
    before_len = len(df_combined)
    df_combined.drop_duplicates(subset=["text"], inplace=True)
    after_len = len(df_combined)
    
    df_combined.to_csv(train_path, index=False, encoding="utf-8")
    print(f"Updated {train_path}: {before_len} -> {after_len} rows (+{len(df_new)} injection samples)")

    # 2. Test 데이터 병합 (validation이 없으면 test를 validation으로 사용하거나 별도 처리)
    # 여기서는 기존 validation.csv가 있다면 거기에 test 셋 일부를 넣거나, 
    # test.csv에 넣는 방식 선택 가능. 요청대로 'data/classification_mod'에 넣음.
    
    test_path = os.path.join(output_dir, "test.csv")
    if os.path.exists(test_path):
        print(f"Reading existing {test_path}...")
        df_test_existing = pd.read_csv(test_path)
    else:
        df_test_existing = pd.DataFrame(columns=["text", "label"])

    print("Processing injection test data...")
    new_test_rows = []
    for row in ds["test"]:
        text = row["prompt"].strip()
        label = row["label"]
        if text:
            if label == 0: #정상
                new_test_rows.append({"text": text, "label": "NORMAL"})
            else:
                new_test_rows.append({"text": text, "label": "INJECTION"})

    df_test_new = pd.DataFrame(new_test_rows)
    df_test_combined = pd.concat([df_test_existing, df_test_new], ignore_index=True)
    df_test_combined.drop_duplicates(subset=["text"], inplace=True)
    
    df_test_combined.to_csv(test_path, index=False, encoding="utf-8")
    print(f"Updated {test_path}: +{len(df_test_new)} injection samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/classification_mod")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    add_injection_data(args.data_dir)


if __name__ == "__main__":
    main()
