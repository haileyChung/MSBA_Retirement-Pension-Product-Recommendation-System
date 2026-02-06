# -*- coding: utf-8 -*-
"""
데이터 로더 모듈
================
포트폴리오 최적화에 필요한 데이터 로딩 클래스

주요 기능:
- 상품 정보 로드
- 시뮬레이션 데이터 로드
- 기대수익률 (Fama-MacBeth) 로드
- 마스크 생성 (시나리오, 위험자산, TDF)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from tqdm import tqdm

from config import PathConfig, OptimizationConfig
from utils import normalize_code


# =============================================================================
# 상품 정보 로더
# =============================================================================

class ProductDataLoader:
    """상품 정보 로딩 클래스"""

    def __init__(self, path_config: PathConfig):
        """
        Args:
            path_config: 경로 설정 객체
        """
        self.path_config = path_config

    def load(self) -> pd.DataFrame:
        """
        상품 정보 로드

        Returns:
            상품 정보 DataFrame (상품 코드, 상품명, 지역, 테마, 투자한도, TDF, 주식채권구분 등)
        """
        df = pd.read_excel(self.path_config.product_info_path)

        # 상품 코드 컬럼 찾기
        code_col = None
        for col in df.columns:
            if '상품' in str(col) and '코드' in str(col):
                code_col = col
                break
        if code_col is None:
            code_col = df.columns[0]

        # 컬럼명 정규화
        df = df.rename(columns={code_col: '상품 코드'})
        df['상품 코드'] = df['상품 코드'].apply(normalize_code)

        print(f"  상품 정보 로드: {len(df)}개")
        return df


# =============================================================================
# 시뮬레이션 데이터 로더
# =============================================================================

class SimulationDataLoader:
    """몬테카를로 시뮬레이션 데이터 로딩 클래스"""

    def __init__(self, path_config: PathConfig, n_simulations: int = 100000):
        """
        Args:
            path_config: 경로 설정 객체
            n_simulations: 예상 시뮬레이션 횟수
        """
        self.path_config = path_config
        self.n_simulations = n_simulations
        self.return_min_threshold = -2.0  # 이상치 하한
        self.return_max_threshold = 5.0   # 이상치 상한

    def load(self, product_codes: List[str]) -> Dict[str, np.ndarray]:
        """
        사전 계산된 누적 수익률 로드

        Args:
            product_codes: 로드할 상품 코드 리스트

        Returns:
            상품코드 -> 누적수익률 배열 매핑 (이상치 제외)
        """
        simulation_data = {}
        missing_codes = []
        outlier_codes = []

        for code in tqdm(product_codes, desc="시뮬레이션 로드"):
            npy_file = os.path.join(self.path_config.simulations_dir, f"{code}_cumulative.npy")

            if os.path.exists(npy_file):
                try:
                    cumulative_returns = np.load(npy_file)

                    if len(cumulative_returns) == self.n_simulations:
                        mean_return = cumulative_returns.mean()
                        if mean_return < self.return_min_threshold or mean_return > self.return_max_threshold:
                            outlier_codes.append((code, mean_return))
                        else:
                            simulation_data[code] = cumulative_returns.astype(np.float64)
                    else:
                        missing_codes.append(code)
                except Exception:
                    missing_codes.append(code)
            else:
                missing_codes.append(code)

        if missing_codes:
            print(f"  ⚠️ 시뮬레이션 없음: {len(missing_codes)}개")

        if outlier_codes:
            print(f"  ⚠️ 이상치 데이터 제외: {len(outlier_codes)}개")

        return simulation_data


# =============================================================================
# 기대수익률 로더 (Fama-MacBeth)
# =============================================================================

class ExpectedReturnLoader:
    """Fama-MacBeth 기대수익률 로딩 클래스"""

    def __init__(self, path_config: PathConfig):
        """
        Args:
            path_config: 경로 설정 객체
        """
        self.path_config = path_config

    def load(self, available_codes: List[str]) -> np.ndarray:
        """
        Fama-MacBeth 기대수익률 로드

        Args:
            available_codes: 사용 가능한 상품 코드 리스트

        Returns:
            기대수익률 배열 (r_hat_annual)
        """
        if not os.path.exists(self.path_config.risk_metrics_path):
            print(f"  ⚠️ risk_metrics.csv 파일 없음: {self.path_config.risk_metrics_path}")
            return None

        df = pd.read_csv(self.path_config.risk_metrics_path, encoding='utf-8-sig')

        if '상품코드' in df.columns:
            df['상품코드'] = df['상품코드'].apply(normalize_code)
            df = df.set_index('상품코드')

        if 'r_hat_annual' not in df.columns:
            print(f"  ⚠️ r_hat_annual 컬럼 없음")
            return None

        expected_returns = []
        missing_codes = []

        for code in available_codes:
            if code in df.index:
                expected_returns.append(df.loc[code, 'r_hat_annual'])
            else:
                expected_returns.append(0.0)
                missing_codes.append(code)

        if missing_codes:
            print(f"  ⚠️ {len(missing_codes)}개 상품 기대수익률 없음 (0으로 대체)")

        expected_returns = np.array(expected_returns, dtype=np.float64)

        print(f"  ✓ Fama-MacBeth 기대수익률 로드: {len(available_codes)}개")
        print(f"    초과수익률 평균: {np.mean(expected_returns)*100:.2f}%")

        return expected_returns


# =============================================================================
# 마스크 생성 클래스
# =============================================================================

class MaskGenerator:
    """제약조건용 마스크 생성 클래스"""

    @staticmethod
    def create_scenario_mask(
        df: pd.DataFrame,
        available_codes: List[str],
        preferred_regions: List[str],
        preferred_themes: List[str]
    ) -> np.ndarray:
        """
        시나리오 마스크 생성

        Args:
            df: 상품 정보 DataFrame
            available_codes: 사용 가능한 상품 코드 리스트
            preferred_regions: 선호 지역 리스트
            preferred_themes: 선호 테마 리스트

        Returns:
            시나리오 마스크 (bool 배열)
        """
        # 빈 리스트면 전체 선택
        if len(preferred_regions) == 0:
            preferred_regions = list(df['지역'].dropna().unique())
        if len(preferred_themes) == 0:
            preferred_themes = list(df['테마'].dropna().unique())

        code_to_info = {}
        for _, row in df.iterrows():
            code = str(row['상품 코드'])
            region = row.get('지역', '')
            theme = row.get('테마', '')
            code_to_info[code] = (region, theme)

        scenario_mask = []
        for code in available_codes:
            region, theme = code_to_info.get(code, ('', ''))
            region_match = region in preferred_regions
            theme_match = theme in preferred_themes
            is_scenario = region_match or theme_match
            scenario_mask.append(is_scenario)

        return np.array(scenario_mask, dtype=bool)

    @staticmethod
    def create_risk_asset_mask(df: pd.DataFrame, available_codes: List[str]) -> np.ndarray:
        """
        위험상품 마스크 생성

        Args:
            df: 상품 정보 DataFrame
            available_codes: 사용 가능한 상품 코드 리스트

        Returns:
            위험상품 마스크 (bool 배열)
        """
        code_to_limit = {}
        for _, row in df.iterrows():
            code = str(row['상품 코드'])
            limit = row.get('투자한도', 1.0)
            code_to_limit[code] = limit

        risk_mask = []
        for code in available_codes:
            limit = code_to_limit.get(code, 1.0)
            is_risk = (limit < 1.0)
            risk_mask.append(is_risk)

        return np.array(risk_mask, dtype=bool)

    @staticmethod
    def create_tdf_mask(
        df: pd.DataFrame,
        available_codes: List[str],
        target_retirement_year: int
    ) -> np.ndarray:
        """
        TDF 마스크 생성

        은퇴연도(target_retirement_year) 이하인 TDF 상품만 마킹
        예: 은퇴연도 2045 → TDF 2030, 2035, 2040, 2045 상품은 True, TDF 2050은 False

        Args:
            df: 상품 정보 DataFrame (TDF 컬럼 포함)
            available_codes: 사용 가능한 상품 코드 리스트
            target_retirement_year: 투자자 은퇴 예정 연도

        Returns:
            TDF 마스크 (bool 배열)
        """
        code_to_tdf = {}
        for _, row in df.iterrows():
            code = str(row['상품 코드'])
            tdf_year = row.get('TDF', 0)
            if pd.isna(tdf_year):
                tdf_year = 0
            code_to_tdf[code] = int(tdf_year)

        tdf_mask = []
        for code in available_codes:
            tdf_year = code_to_tdf.get(code, 0)
            # TDF 상품이고 (tdf_year > 0) 은퇴연도 이하인 경우만 True
            is_valid_tdf = (tdf_year > 0) and (tdf_year <= target_retirement_year)
            tdf_mask.append(is_valid_tdf)

        return np.array(tdf_mask, dtype=bool)


# =============================================================================
# 통합 데이터 로더
# =============================================================================

class PortfolioDataLoader:
    """포트폴리오 최적화용 통합 데이터 로더"""

    def __init__(self, path_config: PathConfig, opt_config: OptimizationConfig):
        """
        Args:
            path_config: 경로 설정 객체
            opt_config: 최적화 설정 객체
        """
        self.path_config = path_config
        self.opt_config = opt_config

        self.product_loader = ProductDataLoader(path_config)
        self.simulation_loader = SimulationDataLoader(path_config, opt_config.n_simulations)
        self.expected_return_loader = ExpectedReturnLoader(path_config)
        self.mask_generator = MaskGenerator()

    def load_all(
        self,
        preferred_regions: List[str],
        preferred_themes: List[str],
        target_retirement_year: int
    ) -> Tuple[pd.DataFrame, Dict, List[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        모든 필요한 데이터 로드

        Args:
            preferred_regions: 선호 지역 리스트
            preferred_themes: 선호 테마 리스트
            target_retirement_year: 은퇴 예정 연도

        Returns:
            (상품정보 DataFrame, 시뮬레이션 데이터, 사용가능 코드 리스트, 수익률 행렬,
             기대수익률, 시나리오 마스크, 위험자산 마스크, TDF 마스크)
        """
        print("\n[데이터 로드]")
        print("-" * 60)

        # 1. 상품 정보
        df_products = self.product_loader.load()
        print(f"  전체 상품: {len(df_products)}개")

        # 2. 시뮬레이션 데이터
        all_codes = df_products['상품 코드'].tolist()
        simulation_data = self.simulation_loader.load(all_codes)

        available_codes = list(simulation_data.keys())
        print(f"  시뮬레이션 가능: {len(available_codes)}개")

        # 3. 수익률 행렬 생성
        returns_matrix = np.column_stack([simulation_data[code] for code in available_codes])
        print(f"  Returns matrix: {returns_matrix.shape}")

        # 4. 기대수익률 (Fama-MacBeth)
        expected_return_fm = self.expected_return_loader.load(available_codes)
        if expected_return_fm is None:
            print("  ⚠️ Fallback: 시뮬레이션 평균 사용")
            expected_return_fm = returns_matrix.mean(axis=0)

        # 5. 마스크 생성
        scenario_mask = self.mask_generator.create_scenario_mask(
            df_products, available_codes, preferred_regions, preferred_themes
        )
        risk_mask = self.mask_generator.create_risk_asset_mask(df_products, available_codes)
        tdf_mask = self.mask_generator.create_tdf_mask(df_products, available_codes, target_retirement_year)

        print(f"  시나리오 상품: {np.sum(scenario_mask)}개")
        print(f"  위험상품: {np.sum(risk_mask)}개")
        print(f"  TDF 상품 (은퇴연도 {target_retirement_year} 이하): {np.sum(tdf_mask)}개")

        return (
            df_products,
            simulation_data,
            available_codes,
            returns_matrix,
            expected_return_fm,
            scenario_mask,
            risk_mask,
            tdf_mask
        )
