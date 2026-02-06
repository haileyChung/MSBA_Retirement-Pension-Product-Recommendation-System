# -*- coding: utf-8 -*-
"""
개인맞춤 리포트 생성 설정 파일

투자 리포트 생성 시스템의 경로, API 키, 모델 파라미터를 관리합니다.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()


# ==================================================================================
# 경로 설정
# ==================================================================================

@dataclass
class PathConfig:
    """경로 설정 클래스"""

    # 프로젝트 루트 경로
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)

    # ===== 입력 파일 경로 =====

    @property
    def portfolio_db_path(self) -> Path:
        """포트폴리오 최적화 결과 DB (4단계 출력)"""
        return self.project_root / "4. 포트폴리오최적화" / "output" / "portfolio_results.db"

    @property
    def product_master_path(self) -> Path:
        """상품 마스터 정보 (2단계 출력)"""
        return self.project_root / "2. 데이터전처리" / "output" / "상품명.xlsx"

    # ===== OCR/NER 입력/출력 경로 =====

    input_pdf_folder: Path = field(
        default_factory=lambda: Path(__file__).parent / "input" / "research_reports"
    )

    output_ocr_folder: Path = field(
        default_factory=lambda: Path(__file__).parent / "output" / "ocr_results"
    )

    output_ner_folder: Path = field(
        default_factory=lambda: Path(__file__).parent / "output" / "ner_results"
    )

    output_insights_db: Path = field(
        default_factory=lambda: Path(__file__).parent / "output" / "insights_merged.json"
    )

    # ===== 리포트 출력 경로 =====

    output_reports_folder: Path = field(
        default_factory=lambda: Path(__file__).parent / "output" / "reports"
    )

    # ===== 외부 프로그램 경로 =====

    poppler_path: Optional[Path] = field(
        default=None,
        metadata={"description": "Poppler 바이너리 경로 (PDF → 이미지 변환용)"}
    )


# ==================================================================================
# OCR 설정
# ==================================================================================

@dataclass
class OCRConfig:
    """OCR 엔진 설정"""

    model_name: str = "gpt-4o"  # OCR은 GPT-4o 사용
    max_tokens: int = 15000
    temperature: float = 0.0  # OCR은 정확한 복사가 목적

    # 환각 방지 프롬프트 사용 여부
    use_strict_transcription: bool = True


# ==================================================================================
# NER 설정
# ==================================================================================

@dataclass
class NERConfig:
    """NER 엔진 설정"""

    model_name: str = "gpt-4o"  # NER은 GPT-4o 사용
    max_tokens: int = 15000
    temperature: float = 0.7  # NER은 해석/분석 필요

    # 하드코딩된 분류 기준
    region_choices: List[str] = field(default_factory=lambda: [
        '한국', '미국', '중국', '아시아', '지역기타'
    ])

    asset_types: List[str] = field(default_factory=lambda: [
        '주식', '채권', '기타'
    ])

    theme_choices: List[str] = field(default_factory=lambda: [
        'AI테크', 'ESG', '금', '바이오헬스케어', '반도체',
        '배터리전기차', '소비재', '지수추종_코스피관련', '지수추종_미국',
        '지수추종_지역특화', '지수추종_한국 코스피 외', '배당',
        '미분류 채권', '연금 특화', '리츠', '테마기타'
    ])

    content_types: List[str] = field(default_factory=lambda: [
        'market_strategy', 'sector_view', 'macro_economy',
        'policy_event', 'risk_alert'
    ])


# ==================================================================================
# 문서 검색 설정
# ==================================================================================

@dataclass
class RetrievalConfig:
    """문서 검색 설정"""

    # 검색 결과 개수
    max_insights: int = 10
    max_news: int = 15

    # 유사도 점수 가중치
    region_match_score: float = 30.0
    theme_match_score: float = 30.0
    keyword_match_score: float = 10.0

    # 네이버 뉴스 API 설정 (직접 입력)
    naver_client_id: str = "YOUR NAVER_CLIENT_ID KEY"
    naver_client_secret: str = "YOUR NAVER_CLIENT_SECRET KEY"
    naver_news_display: int = 10
    naver_news_sort: str = "sim"  # sim: 유사도순, date: 날짜순


# ==================================================================================
# 리포트 생성 설정
# ==================================================================================

@dataclass
class ReportGenerationConfig:
    """리포트 생성 설정"""

    model_name: str = "gpt-5.1"  # 리포트 생성은 GPT-5.1 사용
    temperature: float = 0.7

    # 섹션별 글자수 범위
    section1_min: int = 450
    section1_max: int = 550
    section2_min: int = 450
    section2_max: int = 550
    section3_min: int = 950
    section3_max: int = 1050
    section4_min: int = 450
    section4_max: int = 550

    # 섹션별 최대 토큰 수
    section1_max_tokens: int = 5000
    section2_max_tokens: int = 7000
    section3_max_tokens: int = 12000
    section4_max_tokens: int = 6000

    # 요약문 설정
    summary_max_chars: int = 300
    summary_max_tokens: int = 500


# ==================================================================================
# API 설정
# ==================================================================================

@dataclass
class APIConfig:
    """OpenAI API 설정"""

    # 직접 입력 방식: 아래에 API 키를 입력하세요
    openai_api_key: str = "YOUR_API_KEY"


    # API 호출 재시도 설정
    max_retries: int = 3
    retry_delay: float = 1.0  # 초


# ==================================================================================
# 전체 설정 통합
# ==================================================================================

@dataclass
class Config:
    """전체 설정 통합 클래스"""

    path: PathConfig = field(default_factory=PathConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    ner: NERConfig = field(default_factory=NERConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    report: ReportGenerationConfig = field(default_factory=ReportGenerationConfig)
    api: APIConfig = field(default_factory=APIConfig)

    def validate(self) -> bool:
        """설정 유효성 검증"""
        errors = []

        # API 키 확인
        if not self.api.openai_api_key:
            errors.append("OPENAI_API_KEY가 설정되지 않았습니다.")

        if self.retrieval.naver_client_id and not self.retrieval.naver_client_secret:
            errors.append("네이버 API 키가 불완전합니다.")

        # 경로 확인
        if not self.path.input_pdf_folder.exists():
            errors.append(f"입력 폴더가 존재하지 않습니다: {self.path.input_pdf_folder}")

        if errors:
            print("[설정 오류]")
            for error in errors:
                print(f"  - {error}")
            return False

        return True

    def create_output_dirs(self):
        """출력 디렉토리 생성"""
        self.path.output_ocr_folder.mkdir(parents=True, exist_ok=True)
        self.path.output_ner_folder.mkdir(parents=True, exist_ok=True)
        self.path.output_reports_folder.mkdir(parents=True, exist_ok=True)


# ==================================================================================
# 기본 설정 인스턴스
# ==================================================================================

def load_config() -> Config:
    """설정 로드 및 검증"""
    config = Config()

    if not config.validate():
        raise ValueError("설정이 유효하지 않습니다. 위 오류를 확인하세요.")

    config.create_output_dirs()
    return config
