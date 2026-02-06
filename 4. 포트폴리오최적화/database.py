# -*- coding: utf-8 -*-
"""
데이터베이스 관리 모듈
======================
최적화 결과 DB 저장 및 관리

주요 기능:
- DB 스키마 생성
- 최적화 결과 저장
- 실패 케이스 저장
- 중복 확인
"""

import os
import sqlite3
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional

from config import PathConfig, OptimizationConfig
from utils import excess_to_total_return, calculate_breakdown, create_category_mappings
from utils import calculate_equal_weight_var, calculate_equal_weight_return


# =============================================================================
# DB 스키마 생성
# =============================================================================

def create_database(db_path: str) -> None:
    """
    DB 스키마 생성

    Args:
        db_path: 데이터베이스 파일 경로
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS portfolio_results (
        portfolio_id INTEGER PRIMARY KEY,
        grid_combo_hash TEXT UNIQUE,
        combo_region TEXT,
        combo_theme TEXT,
        combo_target_return REAL,
        combo_retirement_year INTEGER,
        combo_target_excess_return REAL,
        created_at TEXT,
        optimization_success INTEGER DEFAULT 1,
        failure_reason TEXT,
        expected_excess_return_pct REAL,
        expected_total_return_pct REAL,
        expected_excess_return_fm_pct REAL,
        expected_total_return_fm_pct REAL,
        risk_free_rate_pct REAL,
        var_95_pct REAL,
        scenario_weight_pct REAL,
        risk_asset_weight_pct REAL,
        tdf_weight_pct REAL,
        scenario_ex_tdf_weight_pct REAL,
        n_active_products INTEGER,
        n_all_products INTEGER,
        best_strategy_idx INTEGER,
        best_strategy_name TEXT,
        optimization_time_sec REAL,
        region_breakdown_json TEXT,
        theme_breakdown_json TEXT,
        asset_type_breakdown_json TEXT,
        vs_equal_weight_var_pct REAL,
        vs_equal_weight_return_pct REAL,
        products_detail_json TEXT,
        full_data_json TEXT
    )
    """)
    conn.commit()
    conn.close()
    print(f"✅ DB 스키마 생성 완료: {db_path}")


# =============================================================================
# 다음 portfolio_id 반환
# =============================================================================

def get_next_portfolio_id(db_path: str) -> int:
    """
    다음 portfolio_id 반환

    Args:
        db_path: 데이터베이스 파일 경로

    Returns:
        다음 portfolio_id
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COALESCE(MAX(portfolio_id), 0) + 1 FROM portfolio_results")
    next_id = cursor.fetchone()[0]
    conn.close()
    return next_id


# =============================================================================
# 결과 저장 함수
# =============================================================================

def save_to_database(
    db_path: str,
    result: Dict,
    df_products: pd.DataFrame,
    available_codes: List[str],
    elapsed_time: float,
    preferred_regions: List[str],
    preferred_themes: List[str],
    target_return: float,
    target_retirement_year: int,
    opt_config: OptimizationConfig,
    scenario_mask: Optional[np.ndarray] = None,
    tdf_mask: Optional[np.ndarray] = None,
    returns_matrix: Optional[np.ndarray] = None,
    expected_return_fm: Optional[np.ndarray] = None
) -> int:
    """
    결과 DB 저장

    Args:
        db_path: 데이터베이스 파일 경로
        result: 최적화 결과 딕셔너리
        df_products: 상품 정보 DataFrame
        available_codes: 사용 가능한 상품 코드 리스트
        elapsed_time: 소요 시간 (초)
        preferred_regions: 선호 지역 리스트
        preferred_themes: 선호 테마 리스트
        target_return: 목표 총 수익률
        target_retirement_year: 은퇴 예정 연도
        opt_config: 최적화 설정 객체
        scenario_mask: 시나리오 마스크 (optional)
        tdf_mask: TDF 마스크 (optional)
        returns_matrix: 수익률 행렬 (optional)
        expected_return_fm: 기대수익률 (optional)

    Returns:
        portfolio_id
    """
    regions_str = '/'.join(preferred_regions) if preferred_regions else 'ALL'
    themes_str = '/'.join(preferred_themes) if preferred_themes else 'ALL'
    grid_combo_hash = f"{regions_str}_{themes_str}_{target_return:.3f}_{target_retirement_year}"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 중복 확인
    cursor.execute("SELECT portfolio_id FROM portfolio_results WHERE grid_combo_hash = ?", (grid_combo_hash,))
    existing = cursor.fetchone()
    if existing:
        conn.close()
        print(f"⏭️  동일 조건 이미 존재 - 건너뜀")
        return existing[0]

    conn.close()

    portfolio_id = get_next_portfolio_id(db_path)
    created_at = datetime.now().isoformat()
    target_excess_return = target_return - opt_config.risk_free_rate

    # 실패 케이스 처리
    if result is None or (isinstance(result, dict) and result.get('status') == 'failed'):
        failure_reason = "No feasible solution"
        if isinstance(result, dict) and result.get('reason'):
            failure_reason = result.get('reason')

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO portfolio_results (
            portfolio_id, grid_combo_hash, combo_region, combo_theme, combo_target_return,
            combo_retirement_year, combo_target_excess_return, created_at,
            optimization_success, failure_reason, risk_free_rate_pct
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (portfolio_id, grid_combo_hash, regions_str, themes_str, target_return,
              target_retirement_year, target_excess_return, created_at,
              False, failure_reason, opt_config.risk_free_rate * 100))
        conn.commit()
        conn.close()
        print(f"❌ 최적화 실패 저장: {portfolio_id}")
        return portfolio_id

    # 성공 케이스 처리
    code_to_name = dict(zip(df_products['상품 코드'].astype(str), df_products['상품명']))

    # 상품 상세 정보
    products_detail = []
    indices = np.where(result['weights'] > 0)[0]
    sorted_idx = indices[np.argsort(result['weights'][indices])[::-1]]

    rank = 1
    for idx in sorted_idx:
        code = available_codes[idx]
        products_detail.append({
            'rank': int(rank),
            'code': str(code),
            'name': str(code_to_name.get(code, 'Unknown')),
            'weight_pct': round(float(result['weights'][idx]) * 100, 10)
        })
        rank += 1

    expected_total_return_sim = excess_to_total_return(result['expected_return'], opt_config.risk_free_rate)
    expected_total_return_fm = excess_to_total_return(result['expected_return_fm'], opt_config.risk_free_rate)

    # TDF 비중 및 TDF 제외 시나리오 비중 계산
    tdf_weight = result.get('tdf_weight', 0.0)
    scenario_weight = result.get('scenario_weight', 0.0)

    if scenario_mask is not None and tdf_mask is not None:
        weights = result['weights']
        scenario_ex_tdf_mask = scenario_mask & (~tdf_mask)
        scenario_ex_tdf_weight = float(np.sum(weights * scenario_ex_tdf_mask))
    else:
        scenario_ex_tdf_weight = max(0.0, scenario_weight - tdf_weight)

    # 카테고리별 breakdown 계산
    weights = result['weights']
    code_to_category = create_category_mappings(df_products)
    region_breakdown = calculate_breakdown(weights, available_codes, code_to_category, 'region')
    theme_breakdown = calculate_breakdown(weights, available_codes, code_to_category, 'theme')
    asset_type_breakdown = calculate_breakdown(weights, available_codes, code_to_category, 'asset_type')

    # 균등배분 대비 VaR 개선율 계산
    if returns_matrix is not None:
        equal_weight_var = calculate_equal_weight_var(returns_matrix)
        portfolio_var = result['var_95']
        vs_equal_weight_var = ((equal_weight_var - portfolio_var) / equal_weight_var) * 100 if equal_weight_var > 0 else 0.0
    else:
        vs_equal_weight_var = 0.0

    # 균등배분 대비 수익률 차이 계산
    if expected_return_fm is not None:
        equal_weight_return = calculate_equal_weight_return(expected_return_fm)
        portfolio_return_fm = result['expected_return_fm']
        vs_equal_weight_return = (portfolio_return_fm - equal_weight_return) * 100
    else:
        vs_equal_weight_return = 0.0

    full_data = {
        'optimization_result': {
            'expected_excess_return': float(result['expected_return']),
            'expected_total_return': float(expected_total_return_sim),
            'expected_excess_return_fm': float(result['expected_return_fm']),
            'expected_total_return_fm': float(expected_total_return_fm),
            'risk_free_rate': float(opt_config.risk_free_rate),
            'var_95': float(result['var_95']),
            'tdf_weight': float(tdf_weight),
            'scenario_ex_tdf_weight': float(scenario_ex_tdf_weight),
            'vs_equal_weight_var': float(vs_equal_weight_var),
            'vs_equal_weight_return': float(vs_equal_weight_return),
            'best_strategy_idx': int(result.get('best_strategy_idx', -1)),
            'best_strategy_name': str(result.get('best_strategy_name', 'Unknown'))
        },
        'breakdown': {
            'region': region_breakdown,
            'theme': theme_breakdown,
            'asset_type': asset_type_breakdown
        },
        'products': products_detail
    }

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO portfolio_results (
        portfolio_id, grid_combo_hash, combo_region, combo_theme, combo_target_return,
        combo_retirement_year, combo_target_excess_return, created_at,
        optimization_success, failure_reason,
        expected_excess_return_pct, expected_total_return_pct,
        expected_excess_return_fm_pct, expected_total_return_fm_pct,
        risk_free_rate_pct,
        var_95_pct, scenario_weight_pct,
        risk_asset_weight_pct, tdf_weight_pct, scenario_ex_tdf_weight_pct,
        n_active_products, n_all_products,
        best_strategy_idx, best_strategy_name, optimization_time_sec,
        region_breakdown_json, theme_breakdown_json, asset_type_breakdown_json,
        vs_equal_weight_var_pct, vs_equal_weight_return_pct,
        products_detail_json, full_data_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        portfolio_id, grid_combo_hash, regions_str, themes_str, float(target_return),
        int(target_retirement_year), float(target_excess_return), created_at,
        True, None,
        float(result['expected_return'] * 100),
        float(expected_total_return_sim * 100),
        float(result['expected_return_fm'] * 100),
        float(expected_total_return_fm * 100),
        float(opt_config.risk_free_rate * 100),
        float(result['var_95'] * 100),
        float(result['scenario_weight'] * 100),
        float(result['risk_weight'] * 100),
        float(tdf_weight * 100),
        float(scenario_ex_tdf_weight * 100),
        int(result['n_active']),
        int(result['n_all']),
        int(result.get('best_strategy_idx', -1)),
        str(result.get('best_strategy_name', 'Unknown')),
        float(elapsed_time),
        json.dumps(region_breakdown, ensure_ascii=False),
        json.dumps(theme_breakdown, ensure_ascii=False),
        json.dumps(asset_type_breakdown, ensure_ascii=False),
        float(vs_equal_weight_var),
        float(vs_equal_weight_return),
        json.dumps(products_detail, ensure_ascii=False),
        json.dumps(full_data, ensure_ascii=False, indent=2)
    ))

    conn.commit()
    conn.close()
    print(f"✅ DB 저장 완료: {portfolio_id}")
    return portfolio_id


# =============================================================================
# 중복 확인 함수
# =============================================================================

def check_existing_result(
    db_path: str,
    preferred_regions: List[str],
    preferred_themes: List[str],
    target_return: float,
    target_retirement_year: int
) -> Optional[int]:
    """
    동일 조건 결과 존재 여부 확인

    Args:
        db_path: 데이터베이스 파일 경로
        preferred_regions: 선호 지역 리스트
        preferred_themes: 선호 테마 리스트
        target_return: 목표 총 수익률
        target_retirement_year: 은퇴 예정 연도

    Returns:
        존재하면 portfolio_id, 없으면 None
    """
    regions_str = '/'.join(preferred_regions) if preferred_regions else 'ALL'
    themes_str = '/'.join(preferred_themes) if preferred_themes else 'ALL'
    grid_combo_hash = f"{regions_str}_{themes_str}_{target_return:.3f}_{target_retirement_year}"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT portfolio_id FROM portfolio_results WHERE grid_combo_hash = ?", (grid_combo_hash,))
    existing = cursor.fetchone()
    conn.close()

    return existing[0] if existing else None
