# 프로젝트 완료 요약

## 프로젝트 개요

**Transformer 기반 팩터 추출 모델을 활용한 AI 자산배분 시스템**

- 미국 섹터 ETF (11개) + 매크로 지표 (10개)
- Transformer 아키텍처로 팩터 추출
- 복합 손실 함수: 예측(MSE) + 랭킹(IC) + 포트폴리오(Sharpe) + 해석성(L1)
- Docker 기반 완전 자동화 환경

## 생성된 파일 목록

### 핵심 코드
```
models/
├── transformer_model.py     # Transformer 아키텍처
│   ├── SectorEncoder        # 섹터 ETF 인코더
│   ├── MacroEncoder         # 매크로 인코더
│   ├── CrossAttentionFusion # 크로스 어텐션
│   ├── FactorExtractor      # 팩터 추출
│   └── PredictionHead       # 예측 헤드
│
└── loss_functions.py        # 복합 손실 함수
    ├── MSELoss              # 예측 손실
    ├── ICLoss               # IC 손실
    ├── SharpeLoss           # Sharpe 손실
    └── L1RegularizationLoss # L1 정규화

utils/
└── data_loader.py           # 데이터 로더
    ├── ETF 데이터 수집 (yfinance)
    ├── FRED 매크로 데이터 수집
    └── 시계열 시퀀스 생성

train.py                     # 학습 스크립트
inference.py                 # 백테스팅 스크립트
run_example.py               # 빠른 시작 예제
```

### Docker 환경
```
Docker 관련:
├── Dockerfile               # GPU 버전
├── Dockerfile.cpu           # CPU 버전 (가벼움)
├── docker-compose.yml       # 서비스 오케스트레이션
├── .dockerignore            # Docker 빌드 제외
├── .env.example             # 환경 변수 템플릿
├── docker_run.sh            # 실행 스크립트
└── install_docker.sh        # Docker 설치 스크립트
```

### 설정 및 문서
```
configs/
└── config.yaml              # 전체 설정 파일

문서:
├── README.md                # 프로젝트 메인 문서
├── QUICKSTART.md            # 빠른 시작 가이드
├── DOCKER_INSTALL.md        # Docker 설치 가이드
├── DOCKER_USAGE.md          # Docker 사용 가이드
└── PROJECT_SUMMARY.md       # 이 파일

기타:
└── requirements.txt         # Python 의존성
```

## 전체 디렉토리 구조

```
AI_asset_allocation/
├── configs/                 # 설정 파일
│   └── config.yaml
├── data/                    # 데이터
│   ├── raw/                 # 원시 데이터
│   └── processed/           # 전처리된 데이터
├── models/                  # 모델 코드
│   ├── __init__.py
│   ├── transformer_model.py
│   └── loss_functions.py
├── utils/                   # 유틸리티
│   ├── __init__.py
│   └── data_loader.py
├── notebooks/               # Jupyter 노트북
├── results/                 # 결과
│   ├── checkpoints/         # 모델 체크포인트
│   ├── figures/             # 시각화
│   └── logs/                # TensorBoard 로그
├── train.py                 # 학습
├── inference.py             # 백테스팅
├── run_example.py           # 예제
├── requirements.txt         # 의존성
├── Dockerfile               # Docker (GPU)
├── Dockerfile.cpu           # Docker (CPU)
├── docker-compose.yml       # Docker Compose
├── docker_run.sh            # 실행 스크립트
├── install_docker.sh        # 설치 스크립트
├── .dockerignore            # Docker 제외
├── .env.example             # 환경 변수
├── README.md                # 메인 문서
├── QUICKSTART.md            # 빠른 시작
├── DOCKER_INSTALL.md        # Docker 설치
├── DOCKER_USAGE.md          # Docker 사용
└── PROJECT_SUMMARY.md       # 요약
```

## 시작하기

### 1. Docker 설치 (최초 1회)

```bash
cd /mnt/c/projects/AI_asset_allocation

# Docker 설치
bash install_docker.sh

# WSL2 재시작 (Windows PowerShell)
wsl --shutdown
```

### 2. 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# FRED API 키 입력
nano .env
```

### 3. 실행

```bash
# 이미지 빌드
bash docker_run.sh build

# 예제 실행
bash docker_run.sh test

# 컨테이너 진입
bash docker_run.sh cpu
```

### 4. 학습 및 백테스팅

```bash
# 학습
python train.py --config configs/config.yaml

# 백테스팅
python inference.py \
    --config configs/config.yaml \
    --checkpoint results/checkpoints/best_model.pt \
    --split test
```

## 주요 기능

### 1. 데이터 자동 수집
- **섹터 ETF**: yfinance로 자동 다운로드
- **매크로 지표**: FRED API로 자동 수집
- **전처리**: 정규화, 결측치 처리, 시퀀스 생성

### 2. Transformer 모델
- **Sector Encoder**: 섹터별 패턴 학습
- **Macro Encoder**: 매크로 환경 인코딩
- **Cross Attention**: 섹터-매크로 상호작용
- **Factor Extraction**: Mean pooling으로 저차원 팩터 추출

### 3. 복합 손실 함수
```python
L = λ1*MSE + λ2*(-IC) + λ3*(-Sharpe) + λ4*L1
```
- **MSE**: 예측 정확도
- **IC**: Information Coefficient (랭킹)
- **Sharpe**: 포트폴리오 성과
- **L1**: 해석 가능성

### 4. 백테스팅
- 포트폴리오 구성 (Top-K)
- 리밸런싱 시뮬레이션
- 성과 지표 계산
- 시각화 (수익률, Drawdown, 가중치 등)

### 5. Docker 완전 자동화
- CPU/GPU 환경 지원
- Jupyter Notebook 통합
- TensorBoard 통합
- 원클릭 실행

## 설정 옵션

`configs/config.yaml`에서 모든 설정 가능:

### 데이터
- 섹터 ETF 목록 (11개 기본)
- 매크로 지표 (10개 기본)
- 데이터 기간 및 빈도
- Lookback window (기본 60)

### 모델
- Transformer 레이어 수 (3-2-2)
- Attention head 수 (8)
- 임베딩 차원 (128)
- 팩터 차원 (64)

### 손실 함수
- MSE 가중치: 1.0
- IC 가중치: 0.5
- Sharpe 가중치: 0.3
- L1 가중치: 0.01

### 학습
- Batch size: 32
- Epochs: 100
- Learning rate: 0.0001
- Early stopping: 15 epochs

### 포트폴리오
- Top-K: 5
- Rebalance frequency: 20일
- Transaction cost: 0.1%

## Docker 명령어 치트시트

```bash
# 실행
bash docker_run.sh cpu          # CPU 컨테이너
bash docker_run.sh gpu          # GPU 컨테이너
bash docker_run.sh jupyter      # Jupyter
bash docker_run.sh tensorboard  # TensorBoard

# 작업
bash docker_run.sh train        # 학습
bash docker_run.sh test         # 테스트

# 관리
bash docker_run.sh build        # 빌드
bash docker_run.sh stop         # 중지
bash docker_run.sh clean        # 삭제

# 직접 실행
docker-compose up -d transformer-cpu
docker-compose exec transformer-cpu bash
docker-compose run --rm transformer-cpu python train.py
```

## 출력 결과

### 학습 중
- TensorBoard: `http://localhost:6006`
- 로그: `results/logs/`
- 체크포인트: `results/checkpoints/`

### 백테스팅 후
- 시각화: `results/figures/`
  - `backtest_test.png`: 수익률, Drawdown 등
  - `factors_test.png`: 팩터 분석
- 데이터: `results/backtest_test_results.npz`

## 성능 지표

모델은 다음 지표로 평가됩니다:

- **Cumulative Return**: 누적 수익률
- **Annualized Return**: 연환산 수익률
- **Volatility**: 변동성
- **Sharpe Ratio**: 위험 대비 수익
- **Max Drawdown**: 최대 손실
- **Win Rate**: 승률
- **Information Coefficient**: 예측 정확도

## 커스터마이징

### ETF 추가
```yaml
# configs/config.yaml
sector_etfs:
  - XLK
  - NEW_ETF  # 추가
```

### 매크로 지표 추가
```yaml
# configs/config.yaml
macro_indicators:
  - DGS10
  - NEW_INDICATOR  # 추가
```

### 모델 구조 변경
```python
# configs/config.yaml
model:
  sector_encoder:
    num_layers: 4  # 3에서 4로
    d_model: 256   # 128에서 256으로
```

### 손실 가중치 조정
```yaml
# configs/config.yaml
loss:
  ic_weight: 1.0    # IC를 더 중시
  sharpe_weight: 0.5
```

## 기술 스택

- **Python**: 3.10
- **PyTorch**: 2.0+
- **데이터**: yfinance, pandas-datareader, FRED
- **시각화**: matplotlib, seaborn, TensorBoard
- **인프라**: Docker, docker-compose

## 파일 크기

- Docker 이미지: ~3GB (CPU), ~5GB (GPU)
- 프로젝트 코드: ~1MB
- 데이터 (예상): ~50MB
- 모델 체크포인트: ~10-50MB

## 다음 단계

1. **Docker 설치**: `bash install_docker.sh`
2. **FRED API 키 설정**: `.env` 파일 편집
3. **환경 테스트**: `bash docker_run.sh test`
4. **모델 학습**: `bash docker_run.sh train`
5. **결과 확인**: `results/figures/` 디렉토리

## 문서 참고

| 문서 | 설명 |
|------|------|
| [README.md](README.md) | 프로젝트 전체 개요 및 사용법 |
| [QUICKSTART.md](QUICKSTART.md) | 5분 안에 시작하기 |
| [DOCKER_INSTALL.md](DOCKER_INSTALL.md) | Docker 설치 상세 가이드 |
| [DOCKER_USAGE.md](DOCKER_USAGE.md) | Docker 사용법 상세 가이드 |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | 프로젝트 요약 (이 문서) |

## 라이선스

MIT License

## 완료 체크리스트

- ✅ Transformer 모델 구현
- ✅ 데이터 로더 구현
- ✅ 복합 손실 함수 구현
- ✅ 학습 스크립트
- ✅ 백테스팅 스크립트
- ✅ Docker 환경 구성
- ✅ Docker Compose 설정
- ✅ 자동 설치 스크립트
- ✅ 종합 문서화
- ✅ 빠른 시작 가이드

**프로젝트가 완전히 준비되었습니다! 바로 실행 가능합니다.**
