# 개인맞춤 리포트 생성 시스템

Multi-AI Agent 아키텍처를 활용한 퇴직연금 맞춤형 투자 리포트 생성 시스템입니다.

---

## 파일 구조

```
5. 개인맞춤리포트생성/
├── config.py                    # 설정 파일 (경로, API 키, 모델 파라미터)
├── main.py                      # 메인 실행 파일
├── orchestrator.py              # 오케스트레이터 (Agent 조율)
├── ocr_engine.py                # OCR Agent (PDF → 텍스트)
├── ner_engine.py                # NER Agent (텍스트 → 인사이트)
├── document_retrieval.py        # 문서 검색 Agent
├── report_generator.py          # 리포트 생성 Agent (AI Agent 동적 재검색 포함)
├── database.py                  # 포트폴리오 DB 조회
├── utils.py                     # 유틸리티 함수
├── input/                       # 입력 파일 폴더
│   └── research_reports/        # PDF 리서치 리포트
├── output/                      # 결과물 저장 폴더
│   ├── ocr_results/             # OCR 결과 (텍스트)
│   ├── ner_results/             # NER 결과 (JSON)
│   ├── reports/                 # 생성된 리포트
│   └── insights_merged.json     # 병합된 인사이트 DB
├── requirements.txt             # 의존성 패키지
└── README.md                    # README 파일
```

---

## 설치

### 1. 필수 라이브러리 설치

```bash
pip install -r requirements.txt
```

### 2. Poppler 설치 (PDF → 이미지 변환용)

```bash
# macOS
brew install poppler

# Ubuntu/Debian
apt-get install poppler-utils

# Windows
# https://github.com/osber/poppler-windows 에서 다운로드
# 또는 conda install poppler
```

### 3. API 키 설정

`config.py` 파일을 열어 API 키를 직접 입력하세요:

```python
# config.py 192번째 줄 (필수)
openai_api_key: str = "sk-your-api-key-here"

# config.py 141-142번째 줄 (선택)
naver_client_id: str = "your-client-id"
naver_client_secret: str = "your-client-secret"
```

---

## 사전 요구사항

이 폴더의 스크립트를 실행하기 전에 `2. 데이터전처리`와 `4. 포트폴리오최적화` 폴더의 파이프라인을 먼저 실행해야 합니다.

**필요 파일:**
- `2. 데이터전처리/output/상품명.xlsx` (상품 마스터)
- `4. 포트폴리오최적화/output/portfolio_results.db` (포트폴리오 결과)

---

## 사용법

### 메뉴 방식 실행 (권장)

```bash
python main.py
```

메뉴 선택:
- `1`: OCR + NER 파이프라인 실행 (리서치 리포트 분석)
- `2`: 개인맞춤 리포트 생성 (단일 조합)
- `3`: 설정 확인
- `0`: 종료

### Command-line 실행

```bash
# OCR + NER 파이프라인 실행
python main.py ocr-ner

# 리포트 생성 (지역 테마 목표수익률 은퇴연도)
python main.py report 한국 반도체 0.08 2045
```

---

## 시스템 아키텍처

### Multi-AI Agent 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                      Orchestrator (조율자)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐  │
│   │  Agent 1    │   │  Agent 2    │   │     Agent 3         │  │
│   │  DB 적재    │ → │  문서 검색  │ → │   리포트 생성       │  │
│   │ (OCR+NER)   │   │ (Retrieval) │   │  (4섹션+타임라인)   │  │
│   │  GPT-4o     │   │             │   │     GPT-5.1         │  │
│   └─────────────┘   └─────────────┘   └─────────────────────┘  │
│                                                                 │
│                    ┌────────────────────────────────────────┐   │
│                    │  AI Agent 동적 재검색 (섹션별 수행)    │   │
│                    │  - 검색어 자동 생성                    │   │
│                    │  - 인사이트/뉴스 재검색                │   │
│                    │  - 관련도 기반 필터링                  │   │
│                    └────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 실행 단계

### Phase 1: DB 적재 (OCR + NER Agent)

1. PDF 리서치 리포트를 이미지로 변환
2. GPT Vision으로 텍스트 추출 (환각 방지 프롬프트)
3. GPT로 구조화된 인사이트 추출 (지역/테마/센티멘트)
4. 인사이트 JSON 병합 및 저장

**입력:** `input/research_reports/*.pdf`

**출력:**
- `output/ocr_results/*_ocr.txt`
- `output/ner_results/*_ner.json`
- `output/insights_merged.json`

### Phase 2: 문서 검색 (Retrieval Agent)

1. 지역/테마 기반 유사도 검색
2. 관련 리서치 인사이트 조회
3. 네이버 뉴스 API로 최신 뉴스 수집

**출력:** 관련 인사이트 + 뉴스 기사 리스트

### Phase 3: 리포트 생성 (Report Agent) ⭐ AI Agent 방식

1. 포트폴리오 DB에서 최적화 결과 조회
2. **각 섹션별 AI Agent 동적 재검색:**
   - `generate_search_queries_for_section()`: GPT가 섹션 주제에 맞는 검색어 3~5개 자동 생성
   - `search_insights_with_queries()`: 생성된 검색어로 인사이트 재검색 + 관련도 점수 부여
   - `search_news_with_queries()`: 생성된 검색어로 네이버 뉴스 재검색
3. 4개 섹션 생성:
   - 섹션 1: 포트폴리오 구성 상품 설명
   - 섹션 2: 기대 손실감수수준 및 예상 수익률
   - 섹션 3: 시장 전망
   - 섹션 4: 종합 평가
4. 타임라인 생성 (은퇴까지의 투자 여정)

**출력:** `output/reports/report_{지역}_{테마}_{은퇴연도}.json`

---

## 설정 파일 (config.py)

### PathConfig (경로 설정)

| 속성 | 설명 |
|------|------|
| `portfolio_db_path` | 포트폴리오 최적화 결과 DB |
| `product_master_path` | 상품 마스터 정보 |
| `input_pdf_folder` | PDF 입력 폴더 |
| `output_ocr_folder` | OCR 출력 폴더 |
| `output_ner_folder` | NER 출력 폴더 |
| `output_insights_db` | 병합된 인사이트 JSON |
| `output_reports_folder` | 리포트 출력 폴더 |

### OCRConfig (OCR 설정)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `model_name` | gpt-4o | GPT Vision 모델 (OCR용) |
| `max_tokens` | 128000 | 최대 토큰 수 |
| `temperature` | 0.0 | 온도 (정확한 복사를 위해 0) |

### NERConfig (NER 설정)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `model_name` | gpt-4o | GPT 모델 (NER용) |
| `temperature` | 0.7 | 온도 (해석/분석 필요) |
| `region_choices` | ['한국', '미국', ...] | 허용 지역 목록 |
| `theme_choices` | ['AI테크', '반도체', ...] | 허용 테마 목록 |

### RetrievalConfig (문서 검색 설정)

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `max_insights` | 10 | 인사이트 최대 개수 |
| `max_news` | 15 | 뉴스 최대 개수 |
| `region_match_score` | 30.0 | 지역 매칭 점수 |
| `theme_match_score` | 30.0 | 테마 매칭 점수 |

### ReportGenerationConfig (리포트 생성 설정) ⭐

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `model_name` | **gpt-5.1** | GPT 모델 (리포트 생성용) |
| `temperature` | 0.7 | 온도 |
| `section1_min/max` | 450/550 | 섹션 1 글자수 |
| `section2_min/max` | 450/550 | 섹션 2 글자수 |
| `section3_min/max` | 950/1050 | 섹션 3 글자수 |
| `section4_min/max` | 450/550 | 섹션 4 글자수 |

---

## 주요 클래스

### orchestrator.py
- `ReportOrchestrator`: 3개 Agent를 조율하는 중앙 조율자
  - AI Agent 동적 재검색을 위해 `insights_loader`, `news_loader`를 `ReportGenerator`에 전달

### ocr_engine.py
- `VisionOCR`: GPT Vision 기반 OCR 엔진
- `OCRProcessor`: PDF 배치 처리

### ner_engine.py
- `InsightExtractorNER`: 인사이트 추출 엔진
- `NERProcessor`: OCR 결과 배치 처리

### document_retrieval.py
- `InsightsLoader`: 인사이트 JSON 로더
- `NaverNewsLoader`: 네이버 뉴스 API 래퍼
- `DocumentRetriever`: 통합 검색 클래스

### report_generator.py ⭐
- `ReportGenerator`: 4섹션 + 타임라인 생성
  - `generate_search_queries_for_section()`: AI Agent 검색어 자동 생성
  - `search_insights_with_queries()`: AI Agent 인사이트 재검색
  - `search_news_with_queries()`: AI Agent 뉴스 재검색

### database.py
- `PortfolioDatabase`: 포트폴리오 DB 조회
- `ProductMasterLoader`: 상품 마스터 로더

---

## AI Agent 동적 재검색 상세

리포트 생성 시 각 섹션별로 다음 프로세스가 수행됩니다:

```
1. 섹션 주제 분석
   └── GPT가 "포트폴리오 구성 배경", "수익률 분석" 등 주제 파악

2. 검색어 자동 생성 (generate_search_queries_for_section)
   └── GPT가 주제에 맞는 3~5개 검색어 생성
   └── 예: "미국 반도체 2026년 전망", "S&P500 ETF 수익률 비교"

3. 인사이트 재검색 (search_insights_with_queries)
   └── 생성된 검색어로 인사이트 DB 재검색
   └── 검색어 단어가 summary에 포함되면 관련도 점수 +5
   └── 관련도 순 정렬 후 상위 10개 선택

4. 뉴스 재검색 (search_news_with_queries)
   └── 생성된 검색어 상위 3개로 네이버 뉴스 API 호출
   └── 중복 제거 후 통합

5. 섹션 생성
   └── 업데이트된 인사이트/뉴스 정보가 50자 이상이면 사용
   └── 그렇지 않으면 기존 Phase 2 검색 결과 사용
```

---

## 프롬프트 엔지니어링 특징

### 언어 사용 규칙
- 한국어만 사용 (한자, 중국어, 일본어 금지)
- AI, TDF, ETF 등 금융 전문 용어는 예외 허용

### 상품 개수 금지 규칙
- 상품 "개수" 언급 금지 → "비중(%)"으로만 설명
- "OO개 상품", "총 OO종목" 등 금지
- 상품 수가 많다는 부정적 언급 금지

### 리포트 흐름 컨텍스트
- 각 섹션에 "이 섹션은 4개 중 N번째" 설명 포함
- 이전/다음 섹션과의 역할 분담 명시

### 현대차증권 언급 금지
- "현대차증권 리서치팀", "현대차증권에 따르면" 금지
- "시장 분석에 따르면", "최근 데이터 기준" 등 객관적 표현 사용

---

## 출력 리포트 구조

```json
{
  "summary": "포트폴리오 요약 (300자 이내)",
  "section1": "포트폴리오 구성 상품 설명 (450-550자)",
  "section2": "기대 손실감수수준 및 예상 수익률 (450-550자)",
  "section3": "시장 전망 (950-1050자)",
  "section4": "종합 평가 (450-550자)",
  "timeline": {
    "2026": "전략 · 기대효과",
    "2030": "전략 · 기대효과",
    "...": "..."
  }
}
```

---

## 참고사항

- **OCR 환각 방지**: '전사 기계' 페르소나로 원문 그대로 복사
- **NER 분류 규칙**: Controlled Vocabulary 기반 엄격한 분류
- **비동기 처리**: 리포트 생성은 asyncio 기반 비동기 실행
- **HTML 태그**: 리포트 내 `<strong>`, `<br>` 태그 사용
- **API 비용**: GPT-4o Vision (OCR/NER) + GPT-5.1 (리포트 생성) 사용으로 비용 발생
- **AI Agent 재검색**: 각 섹션 생성 시 추가 API 호출 발생 (검색어 생성용)

---

## 폴더 연결 구조

```
1. 데이터크롤링/
    └── output/
         └── (가격, 배당 데이터)
              ↓
2. 데이터전처리/
    └── output/
         └── 상품명.xlsx
              ↓
3. PCA, Fama-MacBeth, GARCHestimation, Simulation/
    └── output/
         └── (기대수익률, 시뮬레이션)
              ↓
4. 포트폴리오최적화/
    └── output/
         └── portfolio_results.db
              ↓
5. 개인맞춤리포트생성/  ← 현재 폴더
    └── output/
         └── reports/*.json
```

---

## 주의사항

1. **API 키 보안**: `.env` 파일을 `.gitignore`에 추가하세요.
2. **API 비용**: GPT-4o Vision은 이미지당 비용이 발생합니다. GPT-5.1은 리포트 생성에 사용됩니다.
3. **Poppler 필수**: PDF → 이미지 변환에 Poppler가 필요합니다.
4. **네트워크 의존**: OpenAI API, 네이버 API 호출이 필요합니다.
5. **메모리 사용**: 대용량 PDF 처리 시 메모리 사용량에 주의하세요.
6. **AI Agent 재검색**: 섹션당 추가 API 호출이 발생하므로 비용 고려 필요

---

## 변경 이력

### v2.0 (2025-02-04)
- 리포트 생성 모델을 `gpt-4o` → `gpt-5.1`로 변경
- AI Agent 동적 재검색 로직 추가:
  - `generate_search_queries_for_section()`: 섹션별 맞춤 검색어 생성
  - `search_insights_with_queries()`: 인사이트 재검색
  - `search_news_with_queries()`: 뉴스 재검색
- 프롬프트 엔지니어링 강화:
  - 언어 사용 규칙 (한자/외국어 금지)
  - 상품 개수 금지 규칙
  - 리포트 흐름 컨텍스트
  - 현대차증권 언급 금지 규칙
- orchestrator.py에서 `insights_loader`, `news_loader`를 `ReportGenerator`에 전달
