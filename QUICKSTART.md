# 빠른 시작 가이드

이 문서는 Docker를 사용하여 프로젝트를 빠르게 시작하는 방법을 안내합니다.

## 사전 요구사항

- Windows 10/11 with WSL2
- 최소 8GB RAM
- 10GB 여유 디스크 공간

## 5분 안에 시작하기

### 1단계: Docker 설치 (최초 1회)

```bash
cd /mnt/c/projects/AI_asset_allocation

# Docker 설치 스크립트 실행
bash install_docker.sh

# WSL2 재시작 (Windows PowerShell에서)
# PowerShell을 열고 실행:
wsl --shutdown

# WSL2 재시작 후 Docker 확인
docker --version
```

### 2단계: FRED API 키 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일 편집
nano .env
# 또는
vim .env
```

`.env` 파일에 API 키 입력:
```
FRED_API_KEY=당신의_실제_API_키
```

API 키 발급: https://fred.stlouisfed.org/docs/api/api_key.html (무료, 1분 소요)

### 3단계: Docker 이미지 빌드

```bash
# CPU 버전 빌드 (약 5-10분 소요)
bash docker_run.sh build
```

### 4단계: 프로젝트 실행

```bash
# 컨테이너 시작 및 진입
bash docker_run.sh cpu

# 컨테이너 내부에서 예제 실행
python run_example.py
```

## 전체 워크플로우

### 환경 확인

```bash
# 컨테이너 내부에서
python run_example.py
```

출력 예시:
```
====================================
Transformer Factor Model - Quick Start Example
====================================

[1/5] Loading configuration...
✓ Configuration loaded
  - Sector ETFs: 11
  - Macro indicators: 10
  - Lookback window: 60

[2/5] Loading market data...
✓ Data loaded successfully
  - Train samples: XXX
  - Val samples: XXX
  - Test samples: XXX

...
```

### 모델 학습

#### 방법 1: 컨테이너 내부에서
```bash
# 컨테이너 진입
bash docker_run.sh cpu

# 학습 시작
python train.py --config configs/config.yaml
```

#### 방법 2: 외부에서 직접 실행
```bash
# 스크립트 사용
bash docker_run.sh train

# 또는 docker-compose 직접 사용
docker-compose run --rm transformer-cpu python train.py --config configs/config.yaml
```

### TensorBoard 모니터링

새 터미널을 열고:
```bash
# TensorBoard 시작
bash docker_run.sh tensorboard

# 브라우저에서 접속
# http://localhost:6006
```

### 백테스팅

학습이 완료되면:
```bash
# 컨테이너 내부에서
python inference.py \
    --config configs/config.yaml \
    --checkpoint results/checkpoints/best_model.pt \
    --split test

# 또는 외부에서
docker-compose run --rm transformer-cpu python inference.py \
    --config configs/config.yaml \
    --checkpoint results/checkpoints/best_model.pt \
    --split test
```

결과 확인:
```bash
# 호스트 시스템에서
ls results/figures/
# backtest_test.png, factors_test.png 등
```

### Jupyter Notebook 사용

```bash
# Jupyter 시작
bash docker_run.sh jupyter

# 브라우저에서 http://localhost:8888 접속
```

## 주요 명령어 요약

```bash
# 컨테이너 시작
bash docker_run.sh cpu          # CPU 모드
bash docker_run.sh gpu          # GPU 모드 (NVIDIA GPU 필요)

# 서비스 시작
bash docker_run.sh jupyter      # Jupyter Notebook
bash docker_run.sh tensorboard  # TensorBoard

# 작업 실행
bash docker_run.sh train        # 모델 학습
bash docker_run.sh test         # 예제 실행

# 관리
bash docker_run.sh build        # 이미지 빌드
bash docker_run.sh stop         # 컨테이너 중지
bash docker_run.sh clean        # 모두 삭제
```

## 문제 해결

### Docker 명령어를 찾을 수 없음

```bash
# Docker 서비스 시작 (WSL2에서)
sudo service docker start

# 또는 Docker Desktop 실행 (Windows)
```

### 권한 오류

```bash
# docker 그룹에 사용자 추가 후 WSL2 재시작
wsl --shutdown  # PowerShell에서
```

### 포트 충돌

다른 애플리케이션이 포트를 사용 중이면 `docker-compose.yml`에서 포트 변경:
```yaml
ports:
  - "6007:6006"  # 6006 대신 6007 사용
```

### 메모리 부족

Windows에서 `.wslconfig` 파일 생성:
```
# C:\Users\사용자명\.wslconfig
[wsl2]
memory=8GB
processors=4
```

## 다음 단계

1. **설정 커스터마이징**: `configs/config.yaml` 편집
2. **모델 파라미터 조정**: Transformer 레이어, 헤드 수 등
3. **새로운 ETF 추가**: config.yaml의 sector_etfs 리스트
4. **손실 함수 가중치 조정**: mse_weight, ic_weight 등

## 유용한 팁

### 데이터 확인
```python
# Jupyter 또는 Python 스크립트에서
from utils.data_loader import MarketDataLoader
import yaml

with open('configs/config.yaml') as f:
    config = yaml.safe_load(f)

loader = MarketDataLoader(config)
data = loader.load_all_data()

print(f"Train: {data['train'][0].shape}")
print(f"Val: {data['val'][0].shape}")
print(f"Test: {data['test'][0].shape}")
```

### 모델 구조 확인
```python
from models.transformer_model import TransformerFactorModel

model = TransformerFactorModel(config)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### 중간 결과 저장
학습 중 체크포인트가 `results/checkpoints/`에 자동 저장됩니다.

### 로그 확인
```bash
# 컨테이너 로그
docker-compose logs transformer-cpu

# 실시간 로그
docker-compose logs -f transformer-cpu
```

## 참고 문서

- [DOCKER_INSTALL.md](DOCKER_INSTALL.md) - Docker 설치 상세 가이드
- [DOCKER_USAGE.md](DOCKER_USAGE.md) - Docker 사용법 상세 가이드
- [README.md](README.md) - 프로젝트 전체 문서

## 도움이 필요하신가요?

- Docker 관련: [DOCKER_INSTALL.md](DOCKER_INSTALL.md) 참고
- 사용법: [DOCKER_USAGE.md](DOCKER_USAGE.md) 참고
- 모델 설명: [README.md](README.md) 참고
