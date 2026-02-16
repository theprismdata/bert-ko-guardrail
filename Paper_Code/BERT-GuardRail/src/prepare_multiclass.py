"""3개 한국어 데이터셋을 10-class 통합 분류 데이터셋으로 변환.

데이터셋:
  - KoSBi v2: HuggingFace에서 자동 다운로드
  - K-MHaS: GitHub에서 자동 다운로드
  - BEEP!: GitHub에서 자동 다운로드

출력:
  - data/classification/{train,validation,test}.csv  (text, label)
  - data/corpus/all_combined.txt                     (통합 코퍼스)

10-class 레이블 체계:
  0: SAFE       — 정상 발화
  1: ORIGIN     — 출신 지역 차별
  2: PHYSICAL   — 외모/신체/장애 차별
  3: POLITICS   — 정치적 편향
  4: PROFANITY  — 욕설/비속어 (성별 무관)
  5: AGE        — 나이/세대 차별
  6: GENDER     — 성별/성적지향 차별
  7: RACE       — 인종/민족/출신국가 차별
  8: RELIGION   — 종교 차별
  9: SOCIAL     — 사회적 지위/학력/가족 차별
"""

import argparse
import csv
import os
import random
from collections import Counter, defaultdict
from urllib.request import urlretrieve

from datasets import load_dataset

# ──────────────────────────────────────────────────────────────
# 10-class 레이블 정의
# ──────────────────────────────────────────────────────────────
LABELS = [
    "SAFE",       # 0
    "ORIGIN",     # 1
    "PHYSICAL",   # 2
    "POLITICS",   # 3
    "PROFANITY",  # 4
    "AGE",        # 5
    "GENDER",     # 6
    "RACE",       # 7
    "RELIGION",   # 8
    "SOCIAL",     # 9
]

# ──────────────────────────────────────────────────────────────
# K-MHaS 라벨 ID → 10-class 매핑
# K-MHaS 원본: 0=origin, 1=physical, 2=politics, 3=profanity,
#              4=age, 5=gender, 6=race, 7=religion, 8=not_hate_speech
# ──────────────────────────────────────────────────────────────
KMHAS_LABEL_MAP = {
    "0": "ORIGIN",
    "1": "PHYSICAL",
    "2": "POLITICS",
    "3": "PROFANITY",
    "4": "AGE",
    "5": "GENDER",
    "6": "RACE",
    "7": "RELIGION",
    "8": "SAFE",
}

KMHAS_URLS = {
    "train": "https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data/kmhas_train.txt",
    "validation": "https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data/kmhas_valid.txt",
    "test": "https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data/kmhas_test.txt",
}

BEEP_URLS = {
    "train": "https://raw.githubusercontent.com/kocohub/korean-hate-speech/master/labeled/train.tsv",
    "test": "https://raw.githubusercontent.com/kocohub/korean-hate-speech/master/labeled/dev.tsv",
}

# ──────────────────────────────────────────────────────────────
# KoSBi v2 demographic_category → 10-class 매핑
# ──────────────────────────────────────────────────────────────
KOSBI_DEMO_MAP = {
    "출신 지역 (국내)": "ORIGIN",
    "외모/용모/신체적 조건": "PHYSICAL",
    "장애/병력": "PHYSICAL",
    "정치적/신념적 성향": "POLITICS",
    "나이/세대": "AGE",
    "성별": "GENDER",
    "성적지향": "GENDER",
    "임신/출산": "GENDER",
    "인종/민족/출신 국가": "RACE",
    "종교": "RELIGION",
    "혼인 여부": "SOCIAL",
    "가족 형태": "SOCIAL",
    "사회적/경제적 신분": "SOCIAL",
    "학력/대학/전공": "SOCIAL",
    "전과": "SOCIAL",
}


# ──────────────────────────────────────────────────────────────
# 다운로드 유틸리티
# ──────────────────────────────────────────────────────────────
def download_file(url, dest):
    """파일이 없으면 다운로드."""
    if not os.path.exists(dest):
        print(f"  Downloading {os.path.basename(dest)}...")
        urlretrieve(url, dest)
    return dest


# ──────────────────────────────────────────────────────────────
# 데이터셋 로더
# ──────────────────────────────────────────────────────────────
def load_kosbi(cache_dir):
    """KoSBi v2 로드 → 10-class 매핑. split별 리스트 반환."""
    print("Loading KoSBi v2...")
    ds = load_dataset("nayohan/KoSBi-v2", cache_dir=cache_dir)

    split_map = {"train": "train", "valid": "validation", "test": "test"}
    result = defaultdict(list)

    for src_split, dst_split in split_map.items():
        if src_split not in ds:
            continue
        for row in ds[src_split]:
            text = row["sentence"].strip().replace("\n", " ")
            if not text:
                continue
            if row["sentence_label"] == "safe":
                label = "SAFE"
            else:
                demo = row["demographic_category"]
                label = KOSBI_DEMO_MAP.get(demo)
                if label is None:
                    label = "PROFANITY"
            result[dst_split].append((text, label))

    for split, rows in result.items():
        print(f"  KoSBi v2 {split}: {len(rows):,} rows")
    return result


def load_kmhas(cache_dir):
    """K-MHaS 다운로드 + 10-class 매핑. split별 리스트 반환."""
    print("Loading K-MHaS...")
    raw_dir = os.path.join(cache_dir, "kmhas")
    os.makedirs(raw_dir, exist_ok=True)

    result = defaultdict(list)
    for split, url in KMHAS_URLS.items():
        path = download_file(url, os.path.join(raw_dir, f"{split}.txt"))
        with open(path, "r", encoding="utf-8") as f:
            f.readline()  # 헤더 스킵 (document\tlabel)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.rsplit("\t", 1)
                if len(parts) != 2:
                    continue
                text = parts[0].strip().strip('"')
                label_str = parts[1].strip()
                if not text:
                    continue
                # 멀티라벨: 첫 번째 라벨 사용
                first_label = label_str.split(",")[0]
                label = KMHAS_LABEL_MAP.get(first_label)
                if label is None:
                    continue
                result[split].append((text, label))

    for split, rows in result.items():
        print(f"  K-MHaS {split}: {len(rows):,} rows")
    return result


def load_beep(cache_dir):
    """BEEP! 다운로드 + 10-class 매핑. split별 리스트 반환.

    BEEP! 컬럼: comments, contain_gender_bias, bias, hate
    매핑:
      hate=none → SAFE
      hate=hate/offensive + gender_bias=True → GENDER
      hate=hate/offensive + gender_bias=False → PROFANITY
    """
    print("Loading BEEP!...")
    raw_dir = os.path.join(cache_dir, "beep")
    os.makedirs(raw_dir, exist_ok=True)

    result = defaultdict(list)
    for split, url in BEEP_URLS.items():
        path = download_file(url, os.path.join(raw_dir, f"{split}.tsv"))
        with open(path, "r", encoding="utf-8") as f:
            f.readline()  # 헤더 스킵
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 4:
                    continue
                text = parts[0].strip()
                gender_bias = parts[1].strip()
                hate = parts[3].strip()
                if not text:
                    continue

                if hate == "none":
                    label = "SAFE"
                elif gender_bias == "True":
                    label = "GENDER"
                else:
                    label = "PROFANITY"

                result[split].append((text, label))

    for split, rows in result.items():
        print(f"  BEEP! {split}: {len(rows):,} rows")
    return result


# ──────────────────────────────────────────────────────────────
# 유틸리티
# ──────────────────────────────────────────────────────────────
def split_train_validation(rows, val_ratio=0.1, seed=42):
    """train 리스트에서 validation을 분할."""
    random.seed(seed)
    shuffled = list(rows)
    random.shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)
    return shuffled[val_size:], shuffled[:val_size]


def save_csv(rows, path):
    """(text, label) 리스트를 CSV로 저장."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["text", "label"])
        for text, label in rows:
            writer.writerow([text, label])
    return len(rows)


def save_corpus(all_rows, path):
    """모든 split의 텍스트를 코퍼스 파일로 저장."""
    with open(path, "w", encoding="utf-8") as f:
        for text, _ in all_rows:
            text = text.strip()
            if text:
                f.write(text + "\n")
    return len(all_rows)


def print_distribution(rows, name):
    """레이블 분포 출력."""
    counts = Counter(label for _, label in rows)
    total = len(rows)
    print(f"\n{name} ({total:,} rows):")
    for label in LABELS:
        cnt = counts.get(label, 0)
        pct = cnt / total * 100 if total > 0 else 0
        print(f"  {label:12s}: {cnt:>7,} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets and build 10-class unified classification dataset")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Base data directory")
    args = parser.parse_args()

    cache_dir = os.path.join(args.data_dir, "cache")
    hf_cache = os.path.join(cache_dir, "huggingface")
    cls_dir = os.path.join(args.data_dir, "classification")
    corpus_dir = os.path.join(args.data_dir, "corpus")
    os.makedirs(cls_dir, exist_ok=True)
    os.makedirs(corpus_dir, exist_ok=True)

    # 1. 3개 데이터셋 로드 (없으면 자동 다운로드)
    kosbi = load_kosbi(hf_cache)
    kmhas = load_kmhas(cache_dir)
    beep = load_beep(cache_dir)

    # 2. BEEP!은 validation이 없으므로 train에서 10% 분할
    if "train" in beep and "validation" not in beep:
        beep_train, beep_val = split_train_validation(beep["train"])
        beep["train"] = beep_train
        beep["validation"] = beep_val
        print(f"  BEEP! train→validation split: "
              f"train={len(beep_train):,}, val={len(beep_val):,}")

    # 3. split별 통합
    unified = defaultdict(list)
    for dataset in [kosbi, kmhas, beep]:
        for split, rows in dataset.items():
            unified[split].extend(rows)

    # 4. 셔플
    for split in unified:
        random.seed(42)
        random.shuffle(unified[split])

    # 5. CSV 저장
    print("\n" + "=" * 60)
    print("Saving unified 10-class CSVs...")
    for split in ["train", "validation", "test"]:
        if split not in unified:
            continue
        path = os.path.join(cls_dir, f"{split}.csv")
        n = save_csv(unified[split], path)
        print(f"  {split}: {path} ({n:,} rows)")

    # 6. 통합 코퍼스 저장
    all_rows = []
    for split in unified:
        all_rows.extend(unified[split])
    corpus_path = os.path.join(corpus_dir, "all_combined.txt")
    n = save_corpus(all_rows, corpus_path)
    print(f"\nCorpus saved: {corpus_path} ({n:,} lines)")

    # 7. 레이블 분포 출력
    print("\n" + "=" * 60)
    for split in ["train", "validation", "test"]:
        if split in unified:
            print_distribution(unified[split], f"{split}")

    # 전체 합산
    all_rows_flat = []
    for split in unified:
        all_rows_flat.extend(unified[split])
    print_distribution(all_rows_flat, "TOTAL")


if __name__ == "__main__":
    main()
