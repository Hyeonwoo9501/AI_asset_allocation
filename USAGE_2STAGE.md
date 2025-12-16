# 2-Stage Factor-Based Portfolio Strategy

## 개요

이 프로젝트는 2단계로 구성된 팩터 기반 포트폴리오 전략입니다:

**Stage 1**: Transformer로 ETF + 매크로 데이터를 팩터 임베딩으로 변환
**Stage 2**: 팩터 분석 및 포트폴리오 최적화 (변동성 높은 팩터 소거)

---

## 📊 전체 흐름

```
[Stage 1: 임베딩 모델 학습]
25개 ETF + 10개 매크로 지표
         ↓
  Transformer Encoder
         ↓
  Factor Embedding (128차원)
         ↓
    학습 & 저장

[Stage 2: 팩터 분석 & 포트폴리오]
저장된 모델로 팩터 추출
         ↓
팩터별 수익률/변동성 계산
         ↓
좋은 팩터만 선택 (높은 수익 + 낮은 변동성)
         ↓
선택된 팩터로 포트폴리오 최적화
```

---

## 🚀 빠른 시작

### 1. 설정 파일 확인

`configs/config.yaml`에서 자산 목록 확인:

```yaml
data:
  # 11개 섹터 ETF
  sector_etfs:
    - XLK  # Technology
    - XLF  # Financial
    # ... (11개)

  # 14개 추가 ETF
  additional_etfs:
    - IWD   # Value Factor
    - IWF   # Growth Factor
    - SPY   # Market
    - TLT   # Long-term Bond
    - GLD   # Gold
    # ... (14개)

model:
  factor_dim: 128  # 팩터 차원 (25개 ETF → 128차원)
```

### 2. Stage 1: 임베딩 모델 학습

```bash
# FRED API 키 설정
export FRED_API_KEY='your_api_key_here'

# 또는 utils/data_loader.py에서 직접 수정
# Fred(api_key='YOUR_FRED_API_KEY')

# 학습 실행
python train_embedding.py --config configs/config.yaml
```

**학습 결과:**
- `results/checkpoints/best_model.pt` - 최적 모델
- `results/logs/` - TensorBoard 로그

**모니터링:**
```bash
tensorboard --logdir results/logs
# http://localhost:6006
```

### 3. Stage 2: 팩터 분석 & 포트폴리오 구성

```bash
python factor_portfolio.py
```

**출력:**
- `results/factor_analysis/factor_metrics.csv` - 팩터별 수익률/변동성/샤프
- `results/factor_analysis/optimal_portfolio.csv` - 최적 포트폴리오 가중치
- `results/factor_analysis/factor_analysis.png` - 팩터 분석 차트

---

## 📈 팩터 선택 전략

### 팩터별 메트릭 계산

```python
# 1. 팩터-수익률 관계 추정 (선형회귀)
# returns[t] = beta @ factors[t] + epsilon
beta = LinearRegression(factors, returns)  # (n_assets, 128)

# 2. 각 팩터의 수익률 기여도
for k in range(128):
    # k번째 팩터만 로딩된 포트폴리오
    factor_portfolio = beta[:, k] / sum(abs(beta[:, k]))

    # 해당 포트폴리오 수익률
    portfolio_returns = returns @ factor_portfolio

    # 팩터와의 상관관계
    factor_return[k] = correlation(factors[:, k], portfolio_returns)

# 3. 팩터 변동성
factor_volatility[k] = std(factors[:, k])

# 4. 팩터 샤프 비율
factor_sharpe[k] = factor_return[k] / factor_volatility[k]
```

### 팩터 선택 기준

**기본 전략** (`adaptive`):
```python
# 1. 수익률 기여가 양수인 팩터만
selected = (factor_return > 0.0)

# 2. 샤프 비율이 최소 기준 이상
selected &= (factor_sharpe > 0.1)

# 3. 샤프 상위 70%
top_70_pct = argsort(factor_sharpe)[-90:]
selected &= in_array(top_70_pct)

# 예: 128개 중 약 80-90개 팩터 선택
```

**커스터마이징:**
```python
# factor_portfolio.py 수정
selected_indices = analyzer.select_factors(
    min_return=0.05,      # 최소 수익률 기여
    min_sharpe=0.2,       # 최소 샤프 비율
    max_volatility=2.0,   # 최대 변동성
    top_k_pct=0.6         # 상위 60%만
)
```

---

## 💼 포트폴리오 최적화

### 평균-분산 최적화 (Factor Model)

```python
# 선택된 팩터만 사용
beta_filtered = beta[:, selected_factors]  # (25, 90)
factor_cov_filtered = factor_cov[selected_factors, :][:, selected_factors]

# 포트폴리오 리스크
# Var(portfolio) = w^T @ beta @ Σ_factor @ beta^T @ w
portfolio_variance = w.T @ beta @ factor_cov @ beta.T @ w

# 최적화
minimize: portfolio_variance
subject to:
  - sum(w) = 1
  - 0 <= w_i <= 0.25  (최대 25% per asset)
  - w @ expected_returns >= target_return (선택적)
```

### 출력 예시

```
=== Optimized Portfolio ===
Expected Return: 0.0082 (0.82%)
Volatility: 0.0145 (1.45%)
Sharpe Ratio: 0.565
Active positions: 8/25

Top 10 Positions:
     asset  weight  expected_return
0      SPY   0.250            0.012
1      XLK   0.220            0.015
2      GLD   0.180            0.008
3     QUAL   0.150            0.010
4      XLV   0.100            0.007
5      IWF   0.050            0.006
6      TLT   0.030            0.004
7      EFA   0.020            0.003
```

---

## 📊 결과 해석

### 1. 팩터 메트릭 (`factor_metrics.csv`)

```csv
factor_id,return,volatility,sharpe,selected
0,0.023,0.145,0.159,True
1,-0.012,0.234,-0.051,False
2,0.045,0.098,0.459,True
...
127,0.018,0.187,0.096,False
```

- `return`: 팩터의 수익률 기여도 (-1 ~ 1)
- `volatility`: 팩터 값의 시간에 따른 변동성
- `sharpe`: return / volatility
- `selected`: True = 포트폴리오에 사용, False = 제거됨

### 2. 팩터 분석 차트 (`factor_analysis.png`)

4개 서브플롯:
1. **Factor Return Distribution**: 대부분 0 근처, 일부 양/음
2. **Factor Volatility Distribution**: 변동성 분포
3. **Factor Sharpe Distribution**: 샤프 비율 (선택 기준)
4. **Return vs Volatility Scatter**: 우상향 = 좋은 팩터

### 3. 포트폴리오 가중치 (`optimal_portfolio.csv`)

최종 투자 비중

---

## ⚙️ 고급 설정

### 팩터 차원 조정

더 많은 정보 저장:
```yaml
# configs/config.yaml
model:
  factor_dim: 256  # 128 → 256

  prediction:
    hidden_dims: [512, 256]  # 용량 증가
```

### 손실 함수 가중치 조정

```yaml
loss:
  mse_weight: 1.0      # 예측 정확도
  ic_weight: 0.8       # IC 더 중시 (0.5 → 0.8)
  sharpe_weight: 0.3
  l1_weight: 0.01
```

### 포트폴리오 제약 조정

```python
# factor_portfolio.py에서
weights = optimizer.optimize(
    expected_returns=expected_returns,
    max_position=0.20,      # 최대 20% (더 분산)
    min_position=0.02,      # 최소 2% (소액 제거)
    target_return=0.01      # 목표 수익률 1%
)
```

---

## 🔍 팩터 해석 (선택적)

팩터가 무엇을 의미하는지 사후 분석:

```python
import numpy as np
from scipy.stats import pearsonr

# 각 팩터와 알려진 변수의 상관관계
for k in range(128):
    corr_vix, _ = pearsonr(factors[:, k], vix_data)
    corr_rate, _ = pearsonr(factors[:, k], interest_rate)
    corr_dollar, _ = pearsonr(factors[:, k], dollar_index)

    if abs(corr_vix) > 0.7:
        print(f"Factor {k}: Volatility factor (corr={corr_vix:.2f})")
    elif abs(corr_rate) > 0.7:
        print(f"Factor {k}: Interest rate factor (corr={corr_rate:.2f})")
```

---

## 📝 체크리스트

### 데이터 준비
- [ ] FRED API 키 설정
- [ ] config.yaml 확인 (ETF 목록, 날짜 범위)

### Stage 1 (임베딩 학습)
- [ ] `python train_embedding.py` 실행
- [ ] TensorBoard 확인 (수렴 여부)
- [ ] `results/checkpoints/best_model.pt` 존재 확인

### Stage 2 (팩터 분석)
- [ ] `python factor_portfolio.py` 실행
- [ ] `results/factor_analysis/factor_metrics.csv` 확인
- [ ] 선택된 팩터 개수 확인 (70-90개 정도가 적당)
- [ ] `optimal_portfolio.csv` 확인

### 결과 검증
- [ ] 포트폴리오 샤프 비율 > 0.5
- [ ] 변동성 < 시장 변동성
- [ ] 분산 투자 (상위 자산 < 30%)

---

## ❓ FAQ

**Q1: 팩터 차원은 어떻게 정하나요?**
A: 자산 개수의 3-5배 정도. 25개 자산 → 128차원 적당. 더 크면 과적합 위험.

**Q2: 선택되는 팩터가 너무 적어요 (30개 미만)**
A: `factor_portfolio.py`에서 `min_sharpe` 낮추기 (0.1 → 0.05)

**Q3: 선택되는 팩터가 너무 많아요 (110개 이상)**
A: `top_k_pct` 낮추기 (0.7 → 0.5)

**Q4: 팩터의 의미를 알 수 있나요?**
A: 딥러닝 팩터는 추상적이라 직접 해석 어려움. 사후 상관관계 분석만 가능.

**Q5: 백테스팅은 어떻게 하나요?**
A: `factor_portfolio.py`를 시간 순으로 rolling하면서 실행 (구현 예정)

---

## 🎯 다음 단계

1. **백테스팅 추가**: 시간에 따른 포트폴리오 성과 평가
2. **리밸런싱 전략**: 언제 포트폴리오를 조정할지
3. **거래 비용 고려**: 슬리피지, 수수료 반영
4. **리스크 관리**: VaR, 드로우다운 제약 추가

---

## 📚 참고

- Transformer: "Attention Is All You Need" (Vaswani et al., 2017)
- Factor Model: Fama-French 5-Factor Model
- Portfolio Optimization: Markowitz Mean-Variance Framework
