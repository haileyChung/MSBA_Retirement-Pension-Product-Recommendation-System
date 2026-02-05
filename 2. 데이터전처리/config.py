# -*- coding: utf-8 -*-
"""
데이터 전처리 관련 설정 파일
- API key, 경로 및 파라미터 관리
"""

import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class LLMConfig:
    """LLM 상품 분류 설정"""

    # ===== 기준 경로 =====
    _current_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))

    # ===== 입력 파일 =====
    input_sheet_name: str = "Sheet1"
    input_name_col: str = "상품명"
    input_code_col: str = "상품 코드"
    input_asset_col: str = "비고"  # 자산군 정보가 '비고' 컬럼에 있음
    input_date_col: str = "최초기준일자"

    @property
    def input_excel_path(self) -> str:
        """상품명 파일 경로 (output 폴더)"""
        return os.path.join(self._current_dir, "output", "상품명.xlsx")

    # ===== OpenAI 설정 =====
    # API 키는 환경변수 OPENAI_API_KEY로 설정하거나 아래에 직접 입력
    # 환경변수 우선, 없으면 아래 값 사용
    openai_api_key: str = "YOUR_API_KEY"  # 여기에 직접 입력하거나 환경변수 사용
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    batch_size: int = 50
    rate_limit_delay: float = 1.0

    # ===== 캐시 =====
    @property
    def cache_file(self) -> str:
        """캐시 파일 경로 (output/상품구분_LLM 폴더)"""
        return os.path.join(self.output_dir, "_label_cache_free.jsonl")

    # ===== 정제 파라미터 =====
    topn_country: int = 10    # 국가 라벨 상위 N개만 유지
    topn_theme: int = 50      # 테마 라벨 상위 N개만 유지
    min_group_size: int = 10  # 최소 그룹 크기 (미만은 '기타')

    # ===== 출력 폴더 =====
    @property
    def output_dir(self) -> str:
        """LLM 결과 저장 폴더"""
        return os.path.join(self._current_dir, "output", "상품구분_LLM")

    # ===== 출력 파일 =====
    @property
    def output_labeled_free(self) -> str:
        return os.path.join(self.output_dir, "labeled_free.csv")

    @property
    def output_labeled_final(self) -> str:
        return os.path.join(self.output_dir, "labeled_final.csv")

    @property
    def output_frequencies(self) -> str:
        return os.path.join(self.output_dir, "label_frequencies.csv")

    @property
    def output_cross_counts(self) -> str:
        return os.path.join(self.output_dir, "factor_cross_counts.csv")

    @property
    def product_name_file(self) -> str:
        """상품명 파일 경로 (분류 결과 추가용)"""
        return os.path.join(self._current_dir, "output", "상품명.xlsx")


@dataclass
class ReturnConfig:
    """수익률 계산 설정"""

    # ===== 기준 경로 (상대경로 계산용) =====
    _base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _current_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))

    @property
    def crawl_dir(self) -> str:
        """크롤링 폴더 경로"""
        return os.path.join(self._base_dir, "1. 데이터크롤링")

    @property
    def crawl_output_dir(self) -> str:
        """크롤링 output 폴더"""
        return os.path.join(self.crawl_dir, "output")

    @property
    def crawl_input_dir(self) -> str:
        """크롤링 input 폴더"""
        return os.path.join(self.crawl_dir, "input")

    # ===== 가격 데이터 파일명 =====
    file_etf: str = "etf_price_filtered.csv"
    file_reits: str = "reits_price_*.csv"  # 패턴 (가장 최근 파일 사용)
    file_fund: str = "fund_price_merged_*.csv"  # 패턴 (가장 최근 파일 사용)

    # ===== 배당 데이터 파일명 =====
    file_etf_div: str = "etf_dividend_*.xlsx"  # 패턴
    file_fund_div: str = "fund_dividend_*.xlsx"  # 패턴
    file_reits_div: str = "reits_dividend_*.xlsx"  # 패턴

    # ===== 무위험수익률 파일명 =====
    file_rf: str = "무위험수익률_최근15년통합.xlsx"

    # ===== 출력 경로 =====
    output_excess_return: str = "상품별일별초과수익률_분배금반영.csv"
    output_summary: str = "상품별_수익률_요약통계.xlsx"

    @property
    def output_dir(self) -> str:
        """출력 폴더 (config.py 기준)"""
        return os.path.join(self._current_dir, "output")

    # ===== 전체 경로 프로퍼티 =====
    @property
    def path_etf(self) -> str:
        return os.path.join(self.crawl_output_dir, self.file_etf)

    @property
    def path_reits(self) -> str:
        return self._get_latest_file(self.crawl_output_dir, self.file_reits)

    @property
    def path_fund(self) -> str:
        return self._get_latest_file(self.crawl_output_dir, self.file_fund)

    @property
    def path_etf_div(self) -> str:
        return self._get_latest_file(self.crawl_output_dir, self.file_etf_div)

    @property
    def path_fund_div(self) -> str:
        return self._get_latest_file(self.crawl_output_dir, self.file_fund_div)

    @property
    def path_reits_div(self) -> str:
        return self._get_latest_file(self.crawl_output_dir, self.file_reits_div)

    @property
    def path_rf(self) -> str:
        return os.path.join(self.crawl_input_dir, self.file_rf)

    def _get_latest_file(self, directory: str, pattern: str) -> str:
        """패턴에 맞는 가장 최근 파일 반환"""
        import glob
        files = glob.glob(os.path.join(directory, pattern))
        if not files:
            return os.path.join(directory, pattern.replace("*", "YYYYMMDD"))
        return max(files, key=os.path.getmtime)

    # ===== 계산 파라미터 =====
    trading_days_per_year: int = 250
    anomaly_threshold: float = -0.1  # 30일 이동평균 대비 -10% 이하면 이상치
    cutoff_date: str = None  # None이면 필터링 없음 (예: "2016-12-31")
    min_data_points: int = 250       # 최소 데이터 수 (미만 상품 제외)

    # ===== 제외할 상품 코드 =====
    excluded_codes: List[str] = field(default_factory=lambda: [
        # 청산 상품
        '494900', '489570', '491080', '098560', '482870', '448600',
        '476810', '467940', '346000', '464060', 'K55301D99288', 'K55101B46238',
        # FUND데이터 중 배당금 정보 없는 상품
        'K55102DR3007', 'K55102DR3015', 'K55214E69177', 'K55301BU4183',
        'K55301BU3524', 'K55301BU3466', 'K55301BU3417', 'K55301BU3003',
        'K55301BU2989', 'K55301BU2971', 'K55301BU2963', 'K55301BU2922',
        'K55234E01566', 'K55234E01558', 'K55234BW7319', 'K55207BU0392',
        'K55104BW3224', 'K55107BB0255', 'K55301BU3102', 'KR5363AC3714',
        'K55107BU7844', 'K55107BU7851', 'K55107BU7943', 'K55107BU7950',
        'K55229BU6989', 'K55223BU7467',
        # 1일 이상 누락 있는 상품 코드
        'K55101D20684', 'K55101DV6856', 'K55101DX8496', 'K55104BW3232',
        'K55105CJ7725', 'K55105CK8670', 'K55105D34215', 'K55105D41632',
        'K55105D42481', 'K55105D42994', 'K55105D43273', 'K55206CD4221',
        'K55206CY3376', 'K55210DT0786', 'K55210DY9889', 'K55213C51700',
        'K55223BU8700', 'K55223CY8549', 'K55232BX7534', 'K55232CJ5287',
        'K55236CG1996', 'K55239CD6805', 'K55301BU4415', 'K55301C65984',
        'K55301CG2721', 'K55303BT4143', 'K55303BT4176', 'K55363B54964',
        'K55364BU0797', 'K55365DA5698', 'K55368D36602',
        # 설정일 오류
        'K55105BU5980'
    ])

    @property
    def output_excess_path(self) -> str:
        return os.path.join(self.output_dir, self.output_excess_return)

    @property
    def output_summary_path(self) -> str:
        return os.path.join(self.output_dir, self.output_summary)


@dataclass
class ProductNameConfig:
    """상품명 생성 설정"""

    # ===== 기준 경로 =====
    _base_dir: str = field(default_factory=lambda: os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _current_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))

    @property
    def crawl_input_dir(self) -> str:
        """크롤링 input 폴더"""
        return os.path.join(self._base_dir, "1. 데이터크롤링", "input")

    # ===== 입력 파일 =====
    # 수익률 결과 (1_수익률계산.py 결과)
    file_excess_return: str = "상품별일별초과수익률_분배금반영.csv"

    # 투자한도 파일 (크롤링 input에서 가져옴)
    file_etf_code: str = "etf_code_list.xlsx"
    file_fund_code: str = "fund_code_list.xlsx"

    @property
    def input_dir(self) -> str:
        """입력 폴더 (output 폴더)"""
        return os.path.join(self._current_dir, "output")

    @property
    def path_excess_return(self) -> str:
        """수익률 결과 파일 경로"""
        return os.path.join(self.input_dir, self.file_excess_return)

    @property
    def path_etf_code(self) -> str:
        """ETF 코드 목록 경로"""
        return os.path.join(self.crawl_input_dir, self.file_etf_code)

    @property
    def path_fund_code(self) -> str:
        """펀드 코드 목록 경로"""
        return os.path.join(self.crawl_input_dir, self.file_fund_code)

    # ===== 출력 파일 =====
    @property
    def output_file(self) -> str:
        """상품명 파일 경로 (output 폴더)"""
        return os.path.join(self._current_dir, "output", "상품명.xlsx")

    # ===== REITs 기본 투자한도 =====
    reits_default_limit: float = 0.7


@dataclass
class AnalysisPeriodConfig:
    """분석기간 설정"""

    # ===== 기준 경로 =====
    _current_dir: str = field(default_factory=lambda: os.path.dirname(os.path.abspath(__file__)))

    # ===== 입력 파일 =====
    # 초과수익률 파일 (1_수익률계산.py 결과)
    file_excess_return: str = "상품별일별초과수익률_분배금반영.csv"

    # 상품 정보 파일
    file_product_info: str = "상품명.xlsx"

    @property
    def input_dir(self) -> str:
        """입력 폴더 (output 폴더)"""
        return os.path.join(self._current_dir, "output")

    @property
    def path_excess_return(self) -> str:
        """초과수익률 파일 경로"""
        return os.path.join(self.input_dir, self.file_excess_return)

    @property
    def path_product_info(self) -> str:
        """상품 정보 파일 경로 (output 폴더)"""
        return os.path.join(self._current_dir, "output", self.file_product_info)

    # ===== 출력 경로 =====
    @property
    def output_dir(self) -> str:
        """출력 폴더 (config.py 기준)"""
        return os.path.join(self._current_dir, "output", "analysis_period")

    # ===== 분석 파라미터 =====
    trading_days_per_year: int = 250
    max_years: int = 15  # 분석할 최대 기간 (년)

    # ===== 분석 카테고리 =====
    categories: List[str] = field(default_factory=lambda: ['지역', '주식채권구분', '테마'])