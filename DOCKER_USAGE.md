# Docker 사용 가이드

## 빠른 시작

### 1. 환경 변수 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집하여 FRED API 키 입력
nano .env
# 또는
vim .env
```

`.env` 파일 내용:
```
FRED_API_KEY=your_actual_api_key_here
```

### 2. Docker 이미지 빌드

```bash
# CPU 버전 빌드 (권장)
docker-compose build transformer-cpu

# GPU 버전 빌드 (NVIDIA GPU 있는 경우)
docker-compose build transformer-gpu

# 모든 서비스 빌드
docker-compose build
```

### 3. 컨테이너 실행

#### CPU 모드
```bash
# 방법 1: 스크립트 사용
bash docker_run.sh cpu

# 방법 2: docker-compose 직접 사용
docker-compose up -d transformer-cpu
docker-compose exec transformer-cpu bash
```

#### GPU 모드
```bash
# 방법 1: 스크립트 사용
bash docker_run.sh gpu

# 방법 2: docker-compose 직접 사용
docker-compose up -d transformer-gpu
docker-compose exec transformer-gpu bash
```

## 주요 명령어

### 개발 환경

```bash
# CPU 컨테이너 시작 및 진입
bash docker_run.sh cpu

# GPU 컨테이너 시작 및 진입
bash docker_run.sh gpu

# Jupyter Notebook 시작
bash docker_run.sh jupyter
# 접속: http://localhost:8888

# TensorBoard 시작
bash docker_run.sh tensorboard
# 접속: http://localhost:6006

# 모든 컨테이너 중지
bash docker_run.sh stop

# 모든 컨테이너 및 이미지 삭제
bash docker_run.sh clean
```

### 모델 학습

```bash
# 컨테이너 내부에서
python train.py --config configs/config.yaml

# 또는 외부에서 직접 실행
docker-compose run --rm transformer-cpu python train.py --config configs/config.yaml

# 스크립트 사용
bash docker_run.sh train
```

### 백테스팅

```bash
# 컨테이너 내부에서
python inference.py \
    --config configs/config.yaml \
    --checkpoint results/checkpoints/best_model.pt \
    --split test

# 또는 외부에서 직접 실행
docker-compose run --rm transformer-cpu python inference.py \
    --config configs/config.yaml \
    --checkpoint results/checkpoints/best_model.pt \
    --split test
```

### 예제 실행

```bash
# 빠른 테스트
bash docker_run.sh test

# 또는
docker-compose run --rm transformer-cpu python run_example.py
```

## 컨테이너 관리

### 컨테이너 상태 확인

```bash
# 실행 중인 컨테이너 확인
docker-compose ps

# 로그 확인
docker-compose logs transformer-cpu

# 실시간 로그 확인
docker-compose logs -f transformer-cpu
```

### 데이터 및 결과 관리

컨테이너와 호스트 간 디렉토리가 자동으로 공유됩니다:

- `./data` → `/app/data` (데이터)
- `./results` → `/app/results` (결과, 체크포인트)
- `./notebooks` → `/app/notebooks` (Jupyter 노트북)

컨테이너에서 생성한 파일이 호스트에도 저장되므로, 컨테이너를 삭제해도 데이터는 보존됩니다.

### 컨테이너 재시작

```bash
# 컨테이너 중지
docker-compose stop transformer-cpu

# 컨테이너 시작
docker-compose start transformer-cpu

# 컨테이너 재시작
docker-compose restart transformer-cpu
```

## 고급 사용법

### 1. 커스텀 명령 실행

```bash
# 단일 명령 실행
docker-compose run --rm transformer-cpu python -c "import torch; print(torch.__version__)"

# 여러 명령 실행
docker-compose run --rm transformer-cpu bash -c "
    python run_example.py && \
    python train.py --config configs/config.yaml
"
```

### 2. Jupyter Notebook 사용

```bash
# Jupyter 시작
docker-compose up -d jupyter

# 브라우저에서 http://localhost:8888 접속
# 비밀번호 없이 바로 사용 가능

# Jupyter 중지
docker-compose stop jupyter
```

Jupyter에서 새 노트북 생성:
```python
# 노트북 예제
import sys
sys.path.append('/app')

from utils.data_loader import MarketDataLoader
from models.transformer_model import TransformerFactorModel

# 모델 로드 및 실험...
```

### 3. TensorBoard 모니터링

```bash
# TensorBoard 시작
docker-compose up -d tensorboard

# 브라우저에서 http://localhost:6006 접속

# 학습 중 실시간 모니터링
# 학습 스크립트가 results/logs에 로그를 저장하면 자동으로 표시됨
```

### 4. GPU 사용

NVIDIA GPU가 있는 경우:

```bash
# nvidia-docker 설치 확인
docker run --rm --gpus all nvidia/cuda:12.1.0-base nvidia-smi

# GPU 컨테이너 실행
docker-compose up -d transformer-gpu
docker-compose exec transformer-gpu bash

# 컨테이너 내부에서 GPU 확인
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### 5. 개발 모드

코드를 수정하면서 실시간으로 테스트:

```bash
# 컨테이너 진입 (볼륨 마운트로 코드 실시간 반영)
docker-compose exec transformer-cpu bash

# 컨테이너 내부에서
python run_example.py  # 코드 수정 후 바로 재실행 가능
```

## 문제 해결

### 포트 충돌

```bash
# 포트가 이미 사용 중인 경우
# docker-compose.yml에서 포트 변경

# 예: 6006 → 6007
ports:
  - "6007:6006"
```

### 메모리 부족

```bash
# docker-compose.yml에 메모리 제한 추가
services:
  transformer-cpu:
    deploy:
      resources:
        limits:
          memory: 8G
```

### 권한 문제

```bash
# 컨테이너에서 생성된 파일의 소유권이 root인 경우
# 호스트에서 소유권 변경
sudo chown -R $USER:$USER ./data ./results
```

### 이미지 재빌드

코드나 requirements.txt를 변경한 경우:

```bash
# 캐시 없이 재빌드
docker-compose build --no-cache transformer-cpu

# 또는
bash docker_run.sh build
```

### 컨테이너 완전 초기화

```bash
# 모든 컨테이너, 이미지, 볼륨 삭제
docker-compose down --rmi all -v

# 재빌드
docker-compose build

# 재시작
bash docker_run.sh cpu
```

## 전체 워크플로우 예제

```bash
# 1. 환경 설정
cp .env.example .env
# .env에 FRED_API_KEY 입력

# 2. 이미지 빌드
bash docker_run.sh build

# 3. 예제 실행 (환경 테스트)
bash docker_run.sh test

# 4. TensorBoard 시작
bash docker_run.sh tensorboard

# 5. 학습 시작 (새 터미널)
bash docker_run.sh train

# 6. 학습 완료 후 백테스팅
docker-compose run --rm transformer-cpu python inference.py \
    --config configs/config.yaml \
    --checkpoint results/checkpoints/best_model.pt \
    --split test

# 7. 결과 확인
ls -l results/figures/
```

## 참고 사항

- **볼륨 마운트**: 코드 변경 사항이 실시간으로 컨테이너에 반영됩니다
- **데이터 영속성**: 컨테이너를 삭제해도 data/ 및 results/ 디렉토리는 보존됩니다
- **포트**: TensorBoard(6006), Jupyter(8888)
- **환경 변수**: .env 파일에서 관리

## 유용한 Docker 명령어

```bash
# 디스크 사용량 확인
docker system df

# 사용하지 않는 리소스 정리
docker system prune -a

# 특정 컨테이너 로그
docker logs ai_asset_allocation_cpu

# 실행 중인 프로세스 확인
docker-compose top

# 컨테이너 리소스 사용량 모니터링
docker stats
```
