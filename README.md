# Transformer Factor Model for Asset Allocation

AI 기반 자산배분을 위한 Transformer 팩터 추출 모델입니다. 미국 섹터 ETF와 매크로 지표를 활용하여 예측 팩터를 생성하고, 포트폴리오 최적화를 수행합니다.

## 프로젝트 개요

### 목적
- **입력**: 미국 섹터 ETF 일/월 수익률 시계열, 매크로 지표(FRED 데이터)
- **모델**: 섹터 인코더 + 매크로 인코더 + 크로스어텐션
- **출력**: Mean pooling 팩터 벡터 `f_t`
- **예측**: `r_hat = Θ * f_t` (선형 회귀)
- **로스**: 예측(MSE) + 랭킹(IC) + 포트폴리오(Sharpe) + 해석성(L1)

### 모델 아키텍처

```
Sector ETF Returns (T, N_sector)    Macro Indicators (T, N_macro)
           ↓                                    ↓
    Sector Encoder                       Macro Encoder
    (Transformer)                       (Transformer)
           ↓                                    ↓
           └──────── Cross Attention ──────────┘
                          ↓
                  Factor Extractor
                   (Mean Pooling)
                          ↓
                  Factor Vector f_t
                          ↓
                  Prediction Head
                      (Θ * f_t)
                          ↓
                  Return Predictions
```

## 프로젝트 구조

```
AI_asset_allocation/
├── configs/
│   └── config.yaml              # 설정 파일
├── data/
│   ├── raw/                     # 원시 데이터
│   └── processed/               # 전처리된 데이터
├── models/
│   ├── transformer_model.py     # Transformer 모델 아키텍처
│   └── loss_functions.py        # 복합 손실 함수
├── utils/
│   └── data_loader.py           # 데이터 로더 (ETF + FRED)
├── notebooks/                   # Jupyter 노트북
├── results/
│   ├── checkpoints/             # 모델 체크포인트
│   ├── figures/                 # 시각화 결과
│   └── logs/                    # TensorBoard 로그
├── train.py                     # 학습 스크립트
├── inference.py                 # 추론 및 백테스팅 스크립트
├── requirements.txt             # 의존성 패키지
└── README.md                    # 프로젝트 문서
```

## 설치

### 방법 1: Docker 사용 (권장)

Docker를 사용하면 환경 설정 없이 바로 실행할 수 있습니다.

#### Docker 설치
```bash
# Docker가 없는 경우 설치
bash install_docker.sh

# 설치 후 WSL2 재시작 필요 (Windows PowerShell에서)
wsl --shutdown
```

자세한 설치 방법은 [DOCKER_INSTALL.md](DOCKER_INSTALL.md) 참고

#### 프로젝트 실행
```bash
# 1. 환경 변수 설정
cp .env.example .env
# .env 파일에 FRED_API_KEY 입력

# 2. Docker 이미지 빌드
bash docker_run.sh build

# 3. 컨테이너 실행
bash docker_run.sh cpu

# 4. 컨테이너 내부에서 예제 실행
python run_example.py
```

자세한 사용법은 [DOCKER_USAGE.md](DOCKER_USAGE.md) 참고

### 방법 2: 로컬 설치

#### 1. 저장소 이동
```bash
cd AI_asset_allocation
```

#### 2. 가상환경 생성 (권장)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

#### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. FRED API 키 설정
FRED 매크로 데이터를 사용하려면 API 키가 필요합니다:
1. [FRED 웹사이트](https://fred.stlouisfed.org/docs/api/api_key.html)에서 무료 API 키 발급
2. `utils/data_loader.py` 파일에서 `YOUR_FRED_API_KEY`를 실제 키로 교체

```python
self.fred = Fred(api_key='YOUR_FRED_API_KEY')
```

또는 환경 변수로 설정:
```bash
export FRED_API_KEY='your_api_key_here'
```

## 사용법

### 1. 설정 파일 수정
`configs/config.yaml`에서 원하는 설정을 조정합니다:
- 섹터 ETF 목록
- 매크로 지표 목록
- 데이터 기간 및 빈도
- 모델 하이퍼파라미터
- 손실 함수 가중치

### 2. 모델 학습
```bash
python train.py --config configs/config.yaml
```

학습 중에는 다음이 수행됩니다:
- ETF 및 매크로 데이터 자동 다운로드
- Train/Validation/Test 분할
- TensorBoard 로그 저장
- 최적 모델 체크포인트 저장

### 3. TensorBoard 모니터링
```bash
tensorboard --logdir results/logs
```

### 4. 백테스팅
학습된 모델로 백테스팅을 수행합니다:

```bash
python inference.py \
    --config configs/config.yaml \
    --checkpoint results/checkpoints/best_model.pt \
    --split test
```

결과:
- 누적 수익률 그래프
- Drawdown 분석
- 포트폴리오 가중치 변화
- 수익률 분포
- 팩터 분석

## 주요 기능

### 1. 복합 손실 함수
모델은 4가지 목표를 동시에 최적화합니다:

```python
L_total = λ1*MSE + λ2*(-IC) + λ3*(-Sharpe) + λ4*L1
```

- **MSE Loss**: 예측 정확도
- **IC Loss**: 랭킹 상관관계 (Information Coefficient)
- **Sharpe Loss**: 포트폴리오 샤프 비율
- **L1 Loss**: 팩터 가중치 해석성

### 2. 크로스 어텐션 메커니즘
섹터 수익률이 매크로 지표에 attend하여 컨텍스트를 학습합니다.

### 3. 팩터 추출
시계열을 mean pooling하여 저차원 팩터 벡터를 추출합니다.

### 4. 포트폴리오 구성
- Top-K 자산 롱 포지션
- 리밸런싱 빈도 조정 가능
- 거래 비용 고려

## 평가 지표

모델은 다음 지표로 평가됩니다:

- **Cumulative Return**: 누적 수익률
- **Annualized Return**: 연환산 수익률
- **Volatility**: 변동성
- **Sharpe Ratio**: 샤프 비율
- **Maximum Drawdown**: 최대 손실폭
- **Win Rate**: 승률
- **Information Coefficient (IC)**: 예측 정확도

## 예제 결과

학습 후 다음과 같은 결과를 확인할 수 있습니다:

```
=== Backtest Results ===
cumulative_return: 0.2543
annualized_return: 0.1821
volatility: 0.1456
sharpe_ratio: 1.2507
max_drawdown: -0.0892
win_rate: 0.5632
```

## 커스터마이징

### 새로운 ETF 추가
`configs/config.yaml`의 `sector_etfs` 리스트에 티커를 추가합니다:

```yaml
sector_etfs:
  - XLK
  - XLF
  - YOUR_NEW_ETF
```

### 새로운 매크로 지표 추가
`configs/config.yaml`의 `macro_indicators` 리스트에 FRED 코드를 추가합니다:

```yaml
macro_indicators:
  - DGS10
  - YOUR_NEW_INDICATOR
```

### 모델 구조 변경
`models/transformer_model.py`에서 레이어 수, 헤드 수, 차원 등을 조정합니다.

### 손실 함수 가중치 조정
`configs/config.yaml`의 `loss` 섹션에서 가중치를 조정합니다:

```yaml
loss:
  mse_weight: 1.0
  ic_weight: 0.5
  sharpe_weight: 0.3
  l1_weight: 0.01
```

## 문제 해결

### FRED API 오류
- API 키가 올바르게 설정되었는지 확인
- 인터넷 연결 확인
- FRED 서비스 상태 확인

### CUDA 오류
GPU 메모리가 부족한 경우 batch_size를 줄입니다:

```yaml
training:
  batch_size: 16  # 기본값 32에서 줄임
```

### 데이터 다운로드 실패
yfinance가 일시적으로 실패할 수 있습니다. 재시도하거나 날짜 범위를 조정합니다.

## 향후 개선 사항

- [ ] 추가 자산 클래스 지원 (채권, 원자재 등)
- [ ] 리스크 패리티 포트폴리오 구성
- [ ] 온라인 학습 (incremental learning)
- [ ] 멀티 태스크 학습 (수익률 + 변동성 예측)
- [ ] 해석 가능한 어텐션 시각화
- [ ] 앙상블 모델

## 참고 문헌

- Vaswani et al. (2017) "Attention Is All You Need"
- Gu et al. (2020) "Empirical Asset Pricing via Machine Learning"
- Zhang et al. (2023) "Deep Learning for Portfolio Optimization"

## 라이선스

MIT License

## 문의

프로젝트에 대한 질문이나 제안이 있으시면 이슈를 열어주세요.
