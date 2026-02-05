# 데이터 전처리 파이프라인

ETF, 펀드, REITs 데이터의 수익률 계산, 상품명 생성, 분석기간 설정, 상품 분류를 위한 전처리 파이프라인입니다.

## 파일 구조

```
2. 데이터전처리/
├── config.py              # 설정 파일 (경로, 파라미터)
├── main.py                # 파이프라인 실행 스크립트
├── 1_수익률계산.py        # 수익률 및 초과수익률 계산
├── 2_상품명생성.py        # 상품명.xlsx 생성
├── 3_분석기간 설정.py     # 데이터 포인트 분석 (선택)
├── 4_상품분류_LLM.py      # 상품명 기반 국가/테마 분류 (선택)
├── requirements.txt       # 의존성 패키지
└── README.md             # 이 파일
```

## 설치

```bash
pip install -r requirements.txt
```

## 사용법

### 빠른 시작

```bash
# 필수 단계 실행 (1-2단계)
python main.py

# 전체 단계 실행 (1-4단계)
python main.py --all

# 특정 단계만 실행
python main.py --step 1
```

### 개별 실행

각 스크립트를 개별적으로 실행할 수도 있습니다.

```bash
python 1_수익률계산.py
python 2_상품명생성.py
python "3_분석기간 설정.py"
python 4_상품분류_LLM.py
```

### 설정 파일 수정

`config.py`에서 데이터 경로와 파라미터를 수정합니다.

**ReturnConfig (수익률 계산용)**
- 가격 데이터 경로: `file_etf`, `file_reits`, `file_fund`
- 배당 데이터 경로: `file_etf_div`, `file_fund_div`, `file_reits_div`
- 무위험수익률: `file_rf`
- 출력 경로: `output_dir`

**ProductNameConfig (상품명 생성용)**
- 수익률 결과: `file_excess_return`
- 투자한도 파일: `file_etf_code`, `file_fund_code`
- 출력 파일: `output_file`

**AnalysisPeriodConfig (분석기간 설정용)**
- 초과수익률 파일: `file_excess_return`
- 상품 정보 파일: `file_product_info`
- 출력 경로: `output_dir`

**LLMConfig (상품 분류용)**
- 입력 파일: `input_excel_path`
- OpenAI API 키: 환경변수 `OPENAI_API_KEY` 또는 `openai_api_key`

## 실행 단계

### 1단계: 수익률 계산 (1_수익률계산.py)
1. ETF/펀드/REITs 가격 데이터 로드 및 표준화
2. 이상치 날짜 탐지 및 필터링
3. 분배금 데이터 로드 및 배당락일 변환
4. 총수익률 계산 (분배금 반영)
5. 초과수익률 계산 (무위험수익률 차감)
6. 상품별 요약 통계 생성

**출력 파일:**
- `output/상품별일별초과수익률_분배금반영.csv`
- `output/상품별_수익률_요약통계.xlsx`

### 2단계: 상품명 생성 (2_상품명생성.py)
1. 수익률 결과에서 상품 정보 추출
2. ETF/FUND 투자한도 매칭
3. TDF 플래그 추가

**출력 파일:**
- `상품명.xlsx`

### 3단계: 분석기간 설정 (3_분석기간 설정.py) - 선택
1. 코드 매칭 검증
2. 시장 팩터 데이터 포인트 분석
3. 카테고리별 데이터 분포 시각화

**출력 파일:**
- `output/analysis_period/` 폴더 내 시각화 이미지

### 4단계: 상품 분류 (4_상품분류_LLM.py) - 선택
1. 상품명을 LLM(GPT)에 전달하여 국가/자산유형/테마 분류
2. 동의어/표기 통합 (정규화)
3. 빈도 기반 정제 (상위 N개만 유지)

**출력 파일:**
- `labeled_free.csv`: 자유 라벨 결과
- `labeled_final.csv`: 정제된 최종 라벨
- `label_frequencies.csv`: 라벨 빈도표
- `factor_cross_counts.csv`: 교차 분포

## 주요 클래스

### config.py
- `ReturnConfig`: 수익률 계산 설정
- `ProductNameConfig`: 상품명 생성 설정
- `AnalysisPeriodConfig`: 분석기간 설정
- `LLMConfig`: LLM 상품 분류 설정

### 1_수익률계산.py
- `PriceDataLoader`: 가격 데이터 로더
- `DividendDataLoader`: 배당 데이터 로더
- `AnomalyDateDetector`: 이상치 날짜 탐지
- `ExDividendDateCalculator`: 배당락일 계산
- `ReturnCalculator`: 수익률 계산
- `SummaryStatisticsCalculator`: 요약 통계 계산
- `ReturnCalculationPipeline`: 메인 파이프라인

### 2_상품명생성.py
- `ProductNameGenerator`: 상품명 생성 파이프라인

### 3_분석기간 설정.py
- `AnalysisPeriodChecker`: 분석기간 검증 및 시각화

### 4_상품분류_LLM.py
- `LabelCache`: 캐시 관리
- `OpenAIClient`: LLM API 클라이언트
- `LabelNormalizer`: 라벨 정규화
- `FrequencyRefiner`: 빈도 기반 정제
- `ProductClassifier`: 메인 파이프라인

## 참고사항

- 코드 정규화: 4~5자리 숫자 코드는 6자리로 변환 (앞에 0 추가)
- 수익률 계산: 단순 수익률 사용 (로그 수익률 X)
- 연간 거래일: 250일 기준
- 이상치 기준: 30일 이동평균 대비 -10% 이하
- 최소 데이터: 250일 (1년) 이상인 상품만 포함
