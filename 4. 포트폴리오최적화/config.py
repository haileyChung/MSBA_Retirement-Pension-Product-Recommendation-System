# -*- coding: utf-8 -*-
"""
포트폴리오 최적화 설정 파일
============================
경로 설정 및 최적화 파라미터 관리

주요 기능:
- PathConfig: 이전 단계 Output 파일 경로 참조
- OptimizationConfig: 최적화 알고리즘 파라미터 설정
"""

import os
from dataclasses import dataclass, field
from typing import List


# =============================================================================
# 경로 설정 클래스
# =============================================================================

@dataclass
class PathConfig:
    """이전 단계 Output 파일 경로 참조 설정"""

    # ===== 기준 경로 =====
    _base_dir: str = field(
        default_factory=lambda: os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )
    )
    # → 최종_코드정리_공유/  (config.py에서 두 단계 상위)

    _current_dir: str = field(
        default_factory=lambda: os.path.dirname(os.path.abspath(__file__))
    )
    # → 4. 포트폴리오최적화/

    # -----------------------------------------------------------------
    # (1) 상품명 및 지역/테마/주식채권 구분 파일
    # -----------------------------------------------------------------
    @property
    def product_info_path(self) -> str:
        """2. 데이터전처리/output/상품명.xlsx"""
        return os.path.join(
            self._base_dir, "2. 데이터전처리", "output", "상품명.xlsx"
        )

    # -----------------------------------------------------------------
    # (2) 분배금 반영된 일별 초과수익률 파일
    # -----------------------------------------------------------------
    @property
    def excess_return_path(self) -> str:
        """2. 데이터전처리/output/상품별일별초과수익률_분배금반영.csv"""
        return os.path.join(
            self._base_dir, "2. 데이터전처리", "output", "상품별일별초과수익률_분배금반영.csv"
        )

    # -----------------------------------------------------------------
    # (3) PCA 관련 output 폴더 (risk_metrics.csv, fama_macbeth_results.csv 등)
    # -----------------------------------------------------------------
    @property
    def pca_output_dir(self) -> str:
        """3. PCA, Fama-MacBeth, GARCHestimation, Simulation/output/"""
        return os.path.join(
            self._base_dir, "3. PCA, Fama-MacBeth, GARCHestimation, Simulation", "output"
        )

    @property
    def risk_metrics_path(self) -> str:
        """3단계 output/risk_metrics.csv"""
        return os.path.join(self.pca_output_dir, "risk_metrics.csv")

    @property
    def fama_macbeth_path(self) -> str:
        """3단계 output/fama_macbeth_results.csv"""
        return os.path.join(self.pca_output_dir, "fama_macbeth_results.csv")

    # -----------------------------------------------------------------
    # (4) 시뮬레이션 결과 폴더 (npy/npz 파일들)
    # -----------------------------------------------------------------
    @property
    def simulations_dir(self) -> str:
        """3. PCA, Fama-MacBeth, GARCHestimation, Simulation/output/simulations/"""
        return os.path.join(self.pca_output_dir, "simulations")

    # -----------------------------------------------------------------
    # (5) 4단계 결과 저장 경로
    # -----------------------------------------------------------------
    @property
    def output_dir(self) -> str:
        """4. 포트폴리오최적화/output/"""
        output_path = os.path.join(self._current_dir, "output")
        os.makedirs(output_path, exist_ok=True)
        return output_path

    @property
    def database_path(self) -> str:
        """최적화 결과 DB 파일 경로"""
        return os.path.join(self.output_dir, "portfolio_results.db")


# =============================================================================
# 최적화 파라미터 설정 클래스
# =============================================================================

@dataclass
class OptimizationConfig:
    """포트폴리오 최적화 알고리즘 파라미터"""

    # ===== 무위험 수익률 =====
    risk_free_rate: float = 0.02532  # 무위험 수익률 2.532% (KOFR 11/13 기준)

    # ===== 시뮬레이션 설정 =====
    n_simulations: int = 100000  # 몬테카를로 시뮬레이션 횟수

    # ===== 투자자 선호 설정 (Portfolio Options Combinations용) =====
    region_options: List[str] = field(default_factory=lambda: ['한국', '미국', '중국', '아시아', '지역기타'])
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

    # ===== 제약조건 설정 =====
    scenario_min: float = 0.20  # 시나리오 상품 최소 비중
    risk_asset_max: float = 0.70  # 위험상품 최대 비중
    tdf_min: float = 0.20  # TDF 상품 최소 비중 (은퇴연도 이하)

    # ===== 학습 파라미터 =====
    n_epochs: int = 3000  # 최대 학습 에포크
    learning_rate: float = 0.05  # 초기 학습률
    patience: int = 300  # Early stopping patience
    tolerance: float = 1e-5  # Loss 개선 판정 임계값

    # ===== 페널티 및 Temperature =====
    penalty_strength: float = 1.0  # 제약조건 위반 페널티 강도
    temperature: float = 1.0  # Soft quantile temperature

    # ===== 초기화 전략 설정 =====
    n_fixed_strategies: int = 8  # 고정 초기화 전략 수
    n_sobol_strategies: int = 8  # Sobol 초기화 전략 수

    @property
    def n_restarts(self) -> int:
        """총 초기화 전략 수"""
        return self.n_fixed_strategies + self.n_sobol_strategies


# =============================================================================
# 설정 인스턴스 생성 함수
# =============================================================================

def create_configs():
    """
    설정 인스턴스 생성

    Returns:
        Tuple[PathConfig, OptimizationConfig]: 경로 설정과 최적화 설정
    """
    path_config = PathConfig()
    opt_config = OptimizationConfig()
    return path_config, opt_config


# =============================================================================
# 확인용: 직접 실행시 설정 내용 출력
# =============================================================================

if __name__ == "__main__":
    path_cfg, opt_cfg = create_configs()

    print("=" * 70)
    print("포트폴리오 최적화 설정 확인")
    print("=" * 70)

    print("\n[경로 설정]")
    print("-" * 60)
    paths = {
        "상품명":                  path_cfg.product_info_path,
        "초과수익률":               path_cfg.excess_return_path,
        "Risk Metrics":           path_cfg.risk_metrics_path,
        "Fama-MacBeth 결과":      path_cfg.fama_macbeth_path,
        "시뮬레이션 폴더":          path_cfg.simulations_dir,
        "결과 저장 폴더":           path_cfg.output_dir,
        "데이터베이스":             path_cfg.database_path,
    }

    for label, path in paths.items():
        exists = "✓" if os.path.exists(path) else "✗"
        print(f"  {exists} {label:20s}: {path}")

    print("\n[최적화 파라미터]")
    print("-" * 60)
    print(f"  무위험 수익률:        {opt_cfg.risk_free_rate*100:.3f}%")
    print(f"  시뮬레이션 횟수:       {opt_cfg.n_simulations:,}회")
    print(f"  학습 에포크:          {opt_cfg.n_epochs:,}회")
    print(f"  초기 학습률:          {opt_cfg.learning_rate}")
    print(f"  초기화 전략:    {opt_cfg.n_restarts}개")
    print(f"  제약조건:")
    print(f"    - 시나리오 최소:     {opt_cfg.scenario_min*100:.0f}%")
    print(f"    - 위험상품 최대:     {opt_cfg.risk_asset_max*100:.0f}%")
    print(f"    - TDF 최소:         {opt_cfg.tdf_min*100:.0f}%")

    print("\n[Portfolio Options Combinations 옵션]")
    print("-" * 60)
    print(f"  지역 옵션:           {opt_cfg.region_options}")
    print(f"  테마 옵션 수:         {len(opt_cfg.theme_options)}개")
    print(f"  목표 수익률 옵션 수:   {len(opt_cfg.target_return_options)}개")
    print(f"  은퇴연도 옵션 수:      {len(opt_cfg.target_retirement_year_options)}개")
    print(f"  총 조합 수:          {len(opt_cfg.region_options) * len(opt_cfg.theme_options) * len(opt_cfg.target_return_options) * len(opt_cfg.target_retirement_year_options):,}개")
