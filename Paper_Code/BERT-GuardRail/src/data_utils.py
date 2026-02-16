"""데이터 로딩 및 전처리 유틸리티."""

import os
from pathlib import Path

from datasets import Dataset, load_dataset

# 프로젝트 표준 10-class 순서 (CLAUDE.md와 동일). 알파벳 순 사용 시 SAFE=8 등 인덱스 꼬임 방지.
MULTICLASS_LABEL_ORDER = [
    "SAFE", "ORIGIN", "PHYSICAL", "POLITICS", "PROFANITY",
    "AGE", "GENDER", "RACE", "RELIGION", "SOCIAL", "INJECTION",
]


def load_text_files(data_dir: str, split_ratio: float = 0.05) -> dict:
    """텍스트 파일들을 로드하여 train/validation으로 분할.

    Args:
        data_dir: .txt 파일들이 있는 디렉토리 경로
        split_ratio: validation 비율 (기본 5%)

    Returns:
        {"train": Dataset, "validation": Dataset}
    """
    txt_files = sorted(str(p) for p in Path(data_dir).glob("*.txt"))
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_dir}")

    ds = load_dataset("text", data_files={"train": txt_files})
    ds = ds["train"].train_test_split(test_size=split_ratio, seed=42)
    return {"train": ds["train"], "validation": ds["test"]}


def load_classification_data(data_dir: str, text_col: str = "text", label_col: str = "label"):
    """CSV/TSV 분류 데이터셋 로드.

    data_dir 아래에 train.csv, validation.csv (선택: test.csv)가 있어야 함.
    TSV도 자동 감지.

    Args:
        data_dir: CSV/TSV 파일이 있는 디렉토리
        text_col: 텍스트 컬럼명
        label_col: 레이블 컬럼명

    Returns:
        dict of Dataset
    """
    splits = {}
    for split_name in ["train", "validation", "test"]:
        for ext in [".csv", ".tsv"]:
            path = os.path.join(data_dir, f"{split_name}{ext}")
            if os.path.exists(path):
                sep = "\t" if ext == ".tsv" else ","
                ds = load_dataset("csv", data_files=path, sep=sep)["train"]
                # 필요한 컬럼만 유지하고 이름 정규화
                if text_col != "text":
                    ds = ds.rename_column(text_col, "text")
                if label_col != "label":
                    ds = ds.rename_column(label_col, "label")
                splits[split_name] = ds
                break

    if "train" not in splits:
        raise FileNotFoundError(f"train.csv or train.tsv not found in {data_dir}")

    return splits


def build_label_map(dataset: Dataset, label_col: str = "label") -> dict:
    """레이블을 정수 인덱스로 매핑하는 딕셔너리 생성.

    프로젝트 10-class(SAFE, ORIGIN, ...) 데이터면 CLAUDE.md 표준 순서를 사용하고,
    그 외(이진 분류 등)는 알파벳 순으로 매핑한다.

    Args:
        dataset: 레이블이 문자열인 Dataset
        label_col: 레이블 컬럼명

    Returns:
        {label_string: int_index, ...}
    """
    unique_labels = set(dataset[label_col])
    if unique_labels == set(MULTICLASS_LABEL_ORDER):
        return {label: idx for idx, label in enumerate(MULTICLASS_LABEL_ORDER)}
    return {label: idx for idx, label in enumerate(sorted(unique_labels))}
