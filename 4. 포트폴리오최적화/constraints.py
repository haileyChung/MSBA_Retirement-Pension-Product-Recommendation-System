# -*- coding: utf-8 -*-
"""
제약조건 검증 모듈
==================
포트폴리오 제약조건 검증 함수

주요 기능:
- Hard Constraints 검증 (Fama-MacBeth r_hat 기준)
- 목표 수익률, 시나리오, 위험자산, TDF 비중 검사
"""

import numpy as np
from typing import Dict, Tuple

from config import OptimizationConfig
from utils import excess_to_total_return


# =============================================================================
# 제약조건 검증 함수
# =============================================================================

def check_hard_constraints(
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    expected_return_fm: np.ndarray,
    target_excess_return: float,
    scenario_mask: np.ndarray,
    risk_mask: np.ndarray,
    tdf_mask: np.ndarray,
    opt_config: OptimizationConfig
) -> Tuple[bool, Dict]:
    """
    Hard Constraints 검증 (Fama-MacBeth r_hat 기준)

    Args:
        weights: 포트폴리오 가중치 [n_products]
        returns_matrix: 수익률 행렬 [n_simulations, n_products]
        expected_return_fm: Fama-MacBeth 기대수익률 [n_products]
        target_excess_return: 목표 초과수익률
        scenario_mask: 시나리오 마스크
        risk_mask: 위험자산 마스크
        tdf_mask: TDF 마스크
        opt_config: 최적화 설정 객체

    Returns:
        (is_valid, metrics_dict): 제약조건 만족 여부와 성과 지표
    """
    # 포트폴리오 수익률 계산
    portfolio_returns = returns_matrix @ weights

    # 기대수익률 계산 (시뮬레이션 평균 vs Fama-MacBeth)
    expected_return_sim = np.mean(portfolio_returns)
    expected_return_fm_val = np.sum(weights * expected_return_fm)

    # VaR 95% 계산
    var_95 = -np.percentile(portfolio_returns, 5)

    # 제약조건 비중 계산
    scenario_weight = np.sum(weights * scenario_mask)
    risk_weight = np.sum(weights * risk_mask)
    tdf_weight = np.sum(weights * tdf_mask)

    # 제약조건 하한 (약간의 허용 오차 포함)
    return_lower_bound = target_excess_return - 1e-6

    # Fama-MacBeth r_hat 기준으로 제약조건 검증 (하한만 적용)
    is_return_ok = expected_return_fm_val >= return_lower_bound
    is_scenario_ok = scenario_weight >= opt_config.scenario_min - 1e-3
    is_risk_ok = risk_weight <= opt_config.risk_asset_max + 1e-3
    is_tdf_ok = tdf_weight >= opt_config.tdf_min - 1e-3

    all_ok = (is_return_ok and is_scenario_ok and is_risk_ok and is_tdf_ok)

    metrics = {
        'expected_return': expected_return_sim,
        'expected_return_fm': expected_return_fm_val,
        'expected_total_return': excess_to_total_return(
            expected_return_fm_val, opt_config.risk_free_rate
        ),
        'var_95': var_95,
        'scenario_weight': scenario_weight,
        'risk_weight': risk_weight,
        'tdf_weight': tdf_weight,
        'is_return_ok': is_return_ok,
        'is_scenario_ok': is_scenario_ok,
        'is_risk_ok': is_risk_ok,
        'is_tdf_ok': is_tdf_ok
    }

    return all_ok, metrics


# =============================================================================
# 제약조건 위반 이유 출력
# =============================================================================

def get_constraint_violation_reasons(metrics: Dict) -> list:
    """
    제약조건 위반 이유 리스트 반환

    Args:
        metrics: check_hard_constraints에서 반환한 metrics 딕셔너리

    Returns:
        위반된 제약조건 이름 리스트
    """
    reasons = []
    if not metrics['is_return_ok']:
        reasons.append("Return")
    if not metrics['is_scenario_ok']:
        reasons.append("Scenario")
    if not metrics['is_risk_ok']:
        reasons.append("Risk")
    if not metrics['is_tdf_ok']:
        reasons.append("TDF")
    return reasons
