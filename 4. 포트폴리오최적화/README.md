# 포트폴리오 최적화

VaR 95% 최소화를 목적함수로 하는 제약조건 기반 포트폴리오 최적화 시스템입니다.

## 파일 구조

```
4. 포트폴리오최적화/
├── config.py                    # 설정 파일 (경로, 파라미터)
├── main.py                     # 메인 실행 파일
├── utils.py                     # 유틸리티 함수
├── data_loader.py              # 데이터 로딩 클래스
├── initialization.py           # 초기화 전략
├── loss_functions.py           # 손실 함수
├── constraints.py              # 제약조건 검증
├── database.py                 # DB 저장/관리
├── portfolio_optimizer.py      # 최적화 엔진
├── requirements.txt            # 의존성 패키지
└── README.md                   # 이 파일
```

---

## 설치

```bash
pip install -r requirements.txt
```

**필수 라이브러리:**
- numpy>=1.21.0
- pandas>=1.3.0
- torch>=2.0.0
- scipy>=1.7.0
- tqdm>=4.62.0
- openpyxl>=3.0.0

**GPU 가속 사용 시 (선택):**
```bash
# PyTorch 공식 사이트 참고
# https://pytorch.org/get-started/locally/
```

---

## 사전 요구사항

이 폴더의 스크립트를 실행하기 전에 `2. 데이터전처리`와 `3. PCA, Fama-MacBeth, GARCHestimation, Simulation` 폴더의 파이프라인을 먼저 실행해야 합니다.

**필요 파일:**
- `2. 데이터전처리/output/상품명.xlsx`
- `2. 데이터전처리/output/상품별일별초과수익률_분배금반영.csv`
- `3. PCA.../output/risk_metrics.csv`
- `3. PCA.../output/fama_macbeth_results.csv`
- `3. PCA.../output/simulations/*.npy`

---

## 사용법

### 메뉴 방식 실행 (권장)

```bash
python main.py
```

메뉴 선택:
- `1`: 단일 조합 최적화
- `2`: Portfolio Options Combinations 포트폴리오옵션조합
- `3`: 설정 확인
- `0`: 종료

### Command-line 실행

```bash
# 단일 조합
python main.py single 한국 반도체 0.08 2045

# Portfolio Options Combinations
python main.py grid
```

---

## 설정 파일 (config.py)

### PathConfig (경로 설정)

| 속성 | 설명 |
|------|------|
| `product_info_path` | 상품명.xlsx |
| `excess_return_path` | 초과수익률.csv |
| `risk_metrics_path` | risk_metrics.csv |
| `simulations_dir` | 시뮬레이션 폴더 |
| `database_path` | 최적화 결과 DB |

### OptimizationConfig (최적화 파라미터)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `risk_free_rate` | 0.02532 | 무위험 수익률 |
| `n_simulations` | 100000 | 몬테카를로 시뮬레이션 횟수 |
| `scenario_min` | 0.20 | 시나리오 상품 최소 비중 |
| `risk_asset_max` | 0.70 | 위험상품 최대 비중 |
| `tdf_min` | 0.20 | TDF 상품 최소 비중 |
| `n_epochs` | 3000 | 최대 학습 에포크 |
| `learning_rate` | 0.05 | 초기 학습률 |
| `patience` | 300 | Early stopping patience |
| `n_fixed_strategies` | 8 | 고정 초기화 전략 수 |
| `n_sobol_strategies` | 8 | Sobol 초기화 전략 수 |

### Portfolio Options Combinations 옵션

**DB 저장 조건을 변경하려면 `config.py`의 다음 부분을 수정하세요:**

```python
# ===== 투자자 선호 설정 (Portfolio Options Combinations용) =====
region_options: List[str] = field(default_factory=lambda: [
    '한국', '미국', '중국', '아시아', '지역기타'
])
theme_options: List[str] = field(default_factory=lambda: [
    'AI테크', 'ESG', '금', '바이오헬스케어', '반도체',
    '배터리전기차', '소비재', '지수추종_한국'
])
target_return_options: List[float] = field(default_factory=lambda: [
    0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06, 0.065, 0.07, 0.075,
    0.08, 0.085, 0.09, 0.095, 0.10, 0.105, 0.11, 0.115, 0.12
])
target_retirement_year_options: List[int] = field(default_factory=lambda: [
    2025, 2030, 2035, 2040, 2045, 2050, 2055, 2060
])
```

**조합 수 계산:**
- 현재 설정: 5개 지역 × 7개 테마 × 19개 수익률 × 8개 은퇴연도 = **5,320개 조합**

**설정 예시:**

```python
# 특정 지역만
region_options: List[str] = field(default_factory=lambda: ['한국'])

# 수익률 범위 축소
target_return_options: List[float] = field(default_factory=lambda: [
    0.05, 0.06, 0.07, 0.08, 0.09, 0.10
])

# 주요 은퇴연도만
target_retirement_year_options: List[int] = field(default_factory=lambda: [
    2030, 2040, 2050
])
```

---

## 최적화 프로세스

### 1단계: 데이터 로드
- 상품 정보, 시뮬레이션 데이터, 기대수익률 로드
- 시나리오/위험자산/TDF 마스크 생성

### 2단계: 병렬 최적화
- 16개 초기화 전략 동시 실행
- 전략별 독립적 Adam 업데이트
- Cosine Annealing LR Schedule
- Early stopping (patience=300)

### 3단계: 제약조건 검증 및 선택
- 유효한 후보 중 최저 VaR 전략 선택
- DB에 결과 저장

---

## 초기화 전략

### 고정 전략 (8개)
1. 균등 배분
2. 시나리오 집중
3. Return충족 + VaR
4. 시나리오 + VaR
5. 안전자산 집중
6. 저변동성 집중
7. CVaR상위 집중
8. 최대손실 회피

### Sobol 전략 (8개)
- Sobol 시퀀스 기반 준난수 초기화
- 정규분포 역변환 스케일링
- 21,201 차원 제한 대응 (타일링)

---

## 출력 결과

### 성과 지표
- VaR 95%: 5% 분위수 손실
- 시뮬레이션 평균 수익률
- Fama-MacBeth r_hat 수익률
- 제약조건 비중 (시나리오/위험자산/TDF)

### DB 스키마
결과는 `output/portfolio_results.db`에 저장됩니다.

**주요 컬럼:**
- `portfolio_id`: 포트폴리오 ID
- `grid_combo_hash`: 조합 해시 (중복 방지)
- `combo_region`, `combo_theme`: 지역/테마 조합
- `combo_target_return`, `combo_retirement_year`: 수익률/은퇴연도
- `var_95_pct`: VaR 95%
- `expected_total_return_fm_pct`: 기대수익률
- `products_detail_json`: 상품별 비중
- `full_data_json`: 전체 결과 JSON

---

## 주요 클래스

### PortfolioOptimizer
```python
optimizer = PortfolioOptimizer(opt_config)
result = optimizer.optimize(
    returns_matrix, expected_return_fm, target_excess_return,
    scenario_mask, risk_mask, tdf_mask
)
```

### PortfolioDataLoader
```python
data_loader = PortfolioDataLoader(path_config, opt_config)
(df_products, simulation_data, available_codes, returns_matrix,
 expected_return_fm, scenario_mask, risk_mask, tdf_mask) = \
    data_loader.load_all(preferred_regions, preferred_themes, target_retirement_year)
```

### InitializationStrategy
```python
initializer = InitializationStrategy(opt_config, device)
init_logits = initializer.create_all_initialization_logits(
    n_products, returns_matrix, scenario_mask, risk_mask,
    target_excess_return, expected_return_fm
)
```

---

## 참고사항

- **목적함수**: VaR 95% 최소화 (Soft Quantile)
- **기대수익률**: Fama-MacBeth r_hat (초과수익률)
- **병렬화**: PyTorch (16개 전략 동시 실행)
- **GPU 메모리**: 대규모 시뮬레이션(100,000회) 시 4GB 이상 권장
- **중복 실행**: 동일 조건은 자동으로 건너뜀 (DB hash 기반)
- **제약조건**: 시나리오 ≥20%, 위험상품 ≤70%, TDF ≥20%
- **은퇴연도**: TDF는 은퇴연도 이하 상품만 포함

---

## 폴더 연결 구조

```
1. 데이터크롤링/
    └── output/
         └── (가격, 배당 데이터)
              ↓
2. 데이터전처리/
    └── output/
         └── (초과수익률, 상품명)
              ↓
3. PCA, Fama-MacBeth, GARCHestimation, Simulation/
    └── output/
         └── (기대수익률, 시뮬레이션)
              ↓
4. 포트폴리오최적화/  ← 현재 폴더
    └── output/
         └── portfolio_results.db
```

---
