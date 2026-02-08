# SBERT 프로젝트 (Docker 실행 가이드)

NVIDIA GB10 등 최신 GPU 환경에서 이 프로젝트를 실행하기 위한 Docker 가이드입니다.

## 1. 서버 세팅 (최초 1회)

### Docker 및 NVIDIA Toolkit 설치
Ubuntu 기준입니다.

```bash
# Docker 설치
curl -fsSL https://get.docker.com | sh

# 일반 사용자(prismdata)에게 Docker 실행 권한 부여
# (설정 후 로그아웃 → 재로그인 필수)
sudo usermod -aG docker $USER

# NVIDIA Container Toolkit 설치 (GPU 사용을 위해 필수)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 2. 단계별 실행 (원격 서버에서 바로 실행)

아래 명령어들은 **Docker 컨테이너를 생성 → 패키지 설치 → 스크립트 실행 → 종료 및 삭제**를 한 번에 수행합니다.
서버 터미널에서 복사해서 붙여넣으세요.

### (옵션) 데이서 셋이 없을 경우 다중 분류 데이터 준비 (Multiclass)
```bash
bash scripts/run_prepare_multiclass.sh
```

### 1. 토크나이저 학습
```bash
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c "pip install transformers datasets scikit-learn && bash scripts/run_tokenizer.sh"
```

### 2. 사전학습 (Pretrain)
```bash
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c "pip install transformers datasets scikit-learn accelerate && bash scripts/run_pretrain.sh"
```

### 3. 파인튜닝 (Finetune)
```bash
docker run --gpus all --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c "pip install transformers datasets scikit-learn accelerate && bash scripts/run_finetune.sh"
```



---

## 3. 개발 및 디버깅 (Interactive Mode)

컨테이너 내부에 들어가서 여러 작업을 계속하거나 디버깅하고 싶을 때 사용합니다.

```bash
# 1. 컨테이너 셸 진입
docker run --gpus all -it --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/pytorch:25.01-py3 bash

# 2. (진입 후) 내부에서 실행
# root@container:/workspace#
pip install transformers datasets scikit-learn accelerate
python src/pretrain.py ...
```

---

## 4. 백그라운드 실행 (로그 저장)

오래 걸리는 학습을 돌려놓고 터미널을 끄고 싶을 때 사용합니다.

```bash
# 사전학습 백그라운드 실행 (로그는 pretrain.log에 저장)
nohup docker run --gpus all --rm -v $(pwd):/workspace -w /workspace nvcr.io/nvidia/pytorch:25.01-py3 \
  bash -c "pip install transformers datasets scikit-learn accelerate && bash scripts/run_pretrain.sh" \
  > pretrain.log 2>&1 &

# 로그 확인
tail -f pretrain.log
```
