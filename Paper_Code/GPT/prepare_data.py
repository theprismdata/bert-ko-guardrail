import os
import pandas as pd
import glob

def prepare_gpt_data():
    # 1. 원본 데이터 경로 설정 (classification_v3 폴더)
    # 프로젝트 경로의 data/gpt 폴더에 저장함.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 프로젝트 루트 경로 (SBERT) 계산: Paper_Code/GPT -> Paper_Code -> SBERT
    project_root = os.path.dirname(os.path.dirname(current_dir))
    print(f"Project root directory: {project_root}")

    # 데이터 경로 설정
    data_dir = os.path.join(project_root, "data", "classification_v3")
    output_dir = os.path.join(project_root, "data", "gpt")
    
    # 출력 폴더가 없으면 생성
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "data.txt")

    print(f"Reading CSV files from: {data_dir}")
    
    # 읽을 파일 목록 (train, validation, test 모두 사용)
    csv_files = ["train.csv", "validation.csv", "test.csv"]
    total_lines = 0

    # 2. 파일 처리 및 저장
    with open(output_file, "w", encoding="utf-8") as f_out:
        for file_name in csv_files:
            file_path = os.path.join(data_dir, file_name)
            
            if not os.path.exists(file_path):
                print(f"Warning: File not found - {file_path}")
                continue

            print(f"Processing {file_name}...")
            
            try:
                # pandas로 CSV 읽기 (text 컬럼만 필요)
                df = pd.read_csv(file_path)
                
                if "text" not in df.columns:
                    print(f"Error: 'text' column not found in {file_name}")
                    continue

                # 텍스트만 추출하여 파일에 쓰기
                texts = df["text"].dropna().astype(str).tolist()
                
                for text in texts:
                    # 빈 줄 제거 및 공백 정리
                    clean_text = text.strip()
                    if clean_text:
                        f_out.write(clean_text + "\n")
                        total_lines += 1
                        
                print(f"  Added {len(texts)} lines from {file_name}")
                
            except Exception as e:
                print(f"Error reading {file_name}: {e}")

    print("-" * 30)
    print(f"Successfully created: {output_file}")
    print(f"Total lines: {total_lines}")
    print("-" * 30)
    print("Now run pretrain_gpt.py with this file!")

if __name__ == "__main__":
    prepare_gpt_data()
