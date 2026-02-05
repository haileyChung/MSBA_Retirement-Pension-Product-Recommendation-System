# 데이터 크롤링 시스템

ETF, 리츠, 펀드의 가격 및 배당 데이터를 수집하는 크롤링 시스템입니다.

---

## 폴더 구조

```
data_crawling/
├── main.py                         # 통합 실행 스크립트
├── config.py                       # 설정 파일 (경로, API 키)
├── 1_etf_price_crawler.py          # ETF 가격 크롤러
├── 2_reits_price_crawler.py        # 리츠 가격 크롤러
├── 3_fund_price_crawler.py         # 펀드 가격 크롤러
├── 4_etf_dividend_crawler.py       # ETF 분배금 크롤러
├── 5_reits_dividend_crawler.py     # 리츠 배당 크롤러
├── 6_fund_dividend_crawler.py      # 펀드 분배금 크롤러
├── input/                          # 입력 파일 폴더
│   ├── etf_code_list.xlsx          # ETF 코드 목록 (1번용)
│   ├── reits_code_list.xlsx        # 리츠 코드 목록 (2번용)
│   ├── fund_code_list.xlsx         # 펀드 코드 목록 (3번용)
│   ├── reits_dividend_list.xlsx    # 리츠 배당 목록 (5번용)
│   └── fund_isin_list.xlsx         # 펀드 ISIN 목록 (6번용)
├── output/                         # 결과물 저장 폴더
└── README.md
```

---

## 사전 준비

### 1. 필수 라이브러리 설치

```bash
pip install pandas requests openpyxl
```

### 2. config.py 설정

`config.py` 파일을 열어 아래 항목을 수정하세요:

```python
# KRX API 키 (ETF, 리츠 가격 크롤링용)
KRX_API_KEY = "YOUR_KRX_API_KEY_HERE"
```

### 3. 입력 파일 준비

`input/` 폴더에 아래 파일들을 준비하세요:

| 파일명 | 용도 | 필수 컬럼 |
|--------|------|-----------|
| `etf_code_list.xlsx` | ETF 가격 크롤링 | 코드 |
| `reits_code_list.xlsx` | 리츠 가격 크롤링 | 리츠 코드, 리츠명, 특징 |
| `fund_code_list.xlsx` | 펀드 가격 크롤링 | 펀드명, 펀드코드, 운용사 |
| `reits_dividend_list.xlsx` | 리츠 배당 크롤링 | 상품 코드, 상품명, KOR_SECN_NM |
| `fund_isin_list.xlsx` | 펀드 배당 크롤링 | ISIN코드 |

---

## 실행 방법

### 통합 실행 (권장)

```bash
python main.py
```

메뉴에서 원하는 작업을 선택하세요:
- 1~6: 개별 크롤러 실행
- 7: 가격 크롤러 전체 실행
- 8: 배당 크롤러 전체 실행
- 9: 전체 크롤러 실행

### 개별 실행

```bash
# ETF 가격 크롤러
python 1_etf_price_crawler.py

# 리츠 가격 크롤러
python 2_reits_price_crawler.py

# 펀드 가격 크롤러
python 3_fund_price_crawler.py

# ETF 분배금 크롤러
python 4_etf_dividend_crawler.py

# 리츠 배당 크롤러
python 5_reits_dividend_crawler.py

# 펀드 분배금 크롤러
python 6_fund_dividend_crawler.py
```

---

## 크롤러별 설명

### 가격 데이터 크롤러

| 크롤러 | 데이터 소스 | 수집 데이터 |
|--------|-------------|-------------|
| 1_etf_price_crawler.py | KRX API | ETF 일별 시세 (종가, 시가, 고가, 저가, 거래량 등) |
| 2_reits_price_crawler.py | KRX API | 리츠 일별 시세 (종가, 시가, 고가, 저가, 거래량 등) |
| 3_fund_price_crawler.py | KOFIA API | 펀드 기준가격 (기준가격, 설정원본 등) |

### 배당 데이터 크롤러

| 크롤러 | 데이터 소스 | 수집 데이터 |
|--------|-------------|-------------|
| 4_etf_dividend_crawler.py | SEIBRO | ETF 분배금 (주당분배금, 권리기준일 등) |
| 5_reits_dividend_crawler.py | SEIBRO | 리츠 배당 (주당배당액, 배정기준일 등) |
| 6_fund_dividend_crawler.py | SEIBRO | 펀드 분배금 (주좌당배당액, 기준일자 등) |

---

## 설정 변경

`config.py`에서 아래 항목을 조정할 수 있습니다:

```python
# 가격 데이터 수집 기간 (년)
PRICE_CRAWL_YEARS = 15

# 배당 데이터 조회 시작일 (YYYYMMDD)
DIVIDEND_START_DATE = "20170101"

# API 호출 간격 (초) - 서버 부하 방지
API_DELAY = 0.3

# 펀드 크롤링 배치 크기
FUND_BATCH_SIZE = 100
```

---

## 출력 파일

결과물은 `output/` 폴더에 저장됩니다:

| 크롤러 | 출력 파일명 |
|--------|-------------|
| ETF 가격 | `etf_price_YYYY.csv`, `etf_price_filtered.csv` |
| 리츠 가격 | `reits_price_YYYYMMDD_HHMMSS.csv` |
| 펀드 가격 | `fund_price_batchN_YYYYMMDD_HHMM.csv`, `fund_price_merged_YYYYMMDD_HHMMSS.csv` |
| ETF 분배금 | `etf_dividend_YYYYMMDD.xlsx` |
| 리츠 배당 | `reits_dividend_YYYYMMDD.xlsx` |
| 펀드 분배금 | `fund_dividend_YYYYMMDD.xlsx` |

---

## 주의사항

1. **API 키 보안**: `config.py`의 API 키를 외부에 노출하지 마세요.
2. **서버 부하**: API 호출 간격(`API_DELAY`)을 적절히 설정하세요.
3. **펀드 크롤링**: 대량 수집 시 중간에 중단될 수 있으므로, 배치 단위로 저장됩니다.
4. **네트워크 오류**: 크롤링 중 네트워크 오류 발생 시 해당 건은 건너뛰고 계속 진행됩니다.

---
