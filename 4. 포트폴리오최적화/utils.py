# -*- coding: utf-8 -*-
"""
유틸리티 함수 모음
==================
공통으로 사용되는 유틸리티 함수 및 클래스

주요 기능:
- 상품코드 정규화
- 수익률 변환 함수
- 진행상황 추적 클래스
- 카테고리별 비중 계산
"""

import time
from typing import Dict
import numpy as np
import pandas as pd


# =============================================================================
# 상품코드 정규화
# =============================================================================

def normalize_code(code) -> str:
    """
    상품코드 정규화 함수

    숫자 코드는 6자리로 맞춤 (앞에 0 채움)

    Args:
        code: 상품 코드 (문자열 또는 숫자)

    Returns:
        정규화된 상품 코드
    """
    code_str = str(code).strip()
    if code_str.isdigit():
        return code_str.zfill(6)
    return code_str


# =============================================================================
# 수익률 변환 함수
# =============================================================================

def get_target_excess_return(target_total_return: float, risk_free_rate: float) -> float:
    """
    고객 희망 총 수익률 → 초과수익률 목표로 변환

    Args:
        target_total_return: 목표 총 수익률
        risk_free_rate: 무위험 수익률

    Returns:
        초과수익률 목표
    """
    return target_total_return - risk_free_rate


def excess_to_total_return(excess_return: float, risk_free_rate: float) -> float:
    """
    초과수익률 → 총 수익률로 변환

    Args:
        excess_return: 초과수익률
        risk_free_rate: 무위험 수익률

    Returns:
        총 수익률
    """
    return excess_return + risk_free_rate


# =============================================================================
# 카테고리별 비중 계산
# =============================================================================

def calculate_breakdown(
    weights: np.ndarray,
    available_codes: list,
    code_to_info: Dict,
    category_key: str
) -> Dict[str, float]:
    """
    특정 카테고리별 비중 분포 계산

    Args:
        weights: 포트폴리오 가중치 배열
        available_codes: 상품 코드 리스트
        code_to_info: 상품별 카테고리 정보 딕셔너리
        category_key: 'region', 'theme', 'asset_type' 중 하나

    Returns:
        카테고리별 비중 (%)
    """
    breakdown = {}
    for idx, code in enumerate(available_codes):
        if weights[idx] > 0:  # 0 초과면 모두 포함
            category = code_to_info.get(code, {}).get(category_key, '기타')
            if pd.isna(category) or category == '':
                category = '기타'
            breakdown[category] = breakdown.get(category, 0.0) + weights[idx]

    # 비중을 % 단위로 변환하고 소수점 10자리로 반올림
    breakdown = {k: round(float(v) * 100, 10) for k, v in breakdown.items()}

    # 비중 큰 순서로 정렬
    breakdown = dict(sorted(breakdown.items(), key=lambda x: x[1], reverse=True))

    return breakdown


def create_category_mappings(df: pd.DataFrame) -> Dict:
    """
    각 상품의 지역, 테마, 자산유형 매핑 생성

    Args:
        df: 상품 정보 DataFrame

    Returns:
        각 상품별 카테고리 정보 딕셔너리
    """
    code_to_info = {}
    for _, row in df.iterrows():
        code = str(row['상품 코드'])
        code_to_info[code] = {
            'region': row.get('지역', '기타'),
            'theme': row.get('테마', '기타'),
            'asset_type': row.get('주식채권구분', '기타')
        }

    return code_to_info


# =============================================================================
# 균등배분 포트폴리오 계산
# =============================================================================

def calculate_equal_weight_var(returns_matrix: np.ndarray) -> float:
    """
    균등배분 포트폴리오의 VaR 95% 계산

    Args:
        returns_matrix: [n_simulations, n_products] 수익률 행렬

    Returns:
        균등배분 VaR 95%
    """
    n_products = returns_matrix.shape[1]
    equal_weights = np.ones(n_products) / n_products
    equal_portfolio_returns = returns_matrix @ equal_weights
    equal_var_95 = -np.percentile(equal_portfolio_returns, 5)
    return equal_var_95


def calculate_equal_weight_return(expected_return_fm: np.ndarray) -> float:
    """
    균등배분 포트폴리오의 기대수익률 계산

    Args:
        expected_return_fm: Fama-MacBeth 기대수익률 배열

    Returns:
        균등배분 기대수익률 (단순 평균)
    """
    return np.mean(expected_return_fm)


# =============================================================================
# 진행상황 추적 클래스
# =============================================================================

class ProgressTracker:
    """
    진행상황 추적 및 시간 추정 클래스

    주요 기능:
    - 진행률 계산 및 표시
    - 경과 시간 추적
    - 남은 시간 예측
    """

    def __init__(self, total: int, desc: str = "Processing"):
        """
        Args:
            total: 전체 작업 수
            desc: 작업 설명
        """
        self.total = total
        self.desc = desc
        self.current = 0
        self.start_time = time.time()

    def update(self, n: int = 1) -> None:
        """
        진행상황 업데이트

        Args:
            n: 진행한 작업 수
        """
        self.current += n

    def get_elapsed_time(self) -> float:
        """
        경과 시간 계산

        Returns:
            경과 시간 (초)
        """
        return time.time() - self.start_time

    def get_eta(self) -> float:
        """
        남은 시간 예측 (ETA: Estimated Time to Arrival)

        Returns:
            예상 남은 시간 (초)
        """
        if self.current == 0:
            return 0
        elapsed = self.get_elapsed_time()
        rate = self.current / elapsed
        remaining = self.total - self.current
        return remaining / rate if rate > 0 else 0

    def print_progress(self) -> None:
        """진행상황 출력"""
        progress_pct = (self.current / self.total) * 100 if self.total > 0 else 0
        elapsed = self.get_elapsed_time()
        eta = self.get_eta()

        print(f"  [{self.desc}] {progress_pct:.1f}% ({self.current}/{self.total}) | "
              f"경과: {elapsed:.1f}s | 남은 시간: {eta:.1f}s")

    def __enter__(self):
        """Context manager 진입"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager 종료"""
        elapsed = self.get_elapsed_time()
        print(f"  [{self.desc}] 완료: {self.total}개 | 소요 시간: {elapsed:.1f}s")


# =============================================================================
# 전략 이름 반환
# =============================================================================

def get_strategy_name(strategy_idx: int, n_fixed_strategies: int = 8) -> str:
    """
    초기화 전략 이름 반환

    Args:
        strategy_idx: 전략 인덱스
        n_fixed_strategies: 고정 전략 수

    Returns:
        전략 이름 (한글)
    """
    strategy_names = [
        "균등 배분",
        "시나리오 집중",
        "Return충족 + VaR",
        "시나리오 + VaR",
        "안전자산 집중",
        "저변동성 집중",
        "CVaR상위 집중",
        "최대손실 회피"
    ]

    if strategy_idx < n_fixed_strategies:
        return strategy_names[strategy_idx]
    else:
        return f"Sobol_{strategy_idx - n_fixed_strategies + 1}"
