# -*- coding: utf-8 -*-
"""
초기화 전략 모듈
================
포트폴리오 최적화를 위한 초기화 전략

주요 기능:
- 고정 초기화 전략 (8가지)
- Sobol 시퀀스 기반 랜덤 초기화
- 단일 Sampler 연속 샘플링
"""

import numpy as np
import torch
from scipy.stats import qmc, norm
from typing import Tuple

from config import OptimizationConfig


# =============================================================================
# 초기화 전략 생성 클래스
# =============================================================================

class InitializationStrategy:
    """초기화 전략 생성 클래스"""

    def __init__(self, opt_config: OptimizationConfig, device: torch.device):
        """
        Args:
            opt_config: 최적화 설정 객체
            device: 디바이스
        """
        self.opt_config = opt_config
        self.device = device
        self.n_fixed = opt_config.n_fixed_strategies
        self.n_sobol = opt_config.n_sobol_strategies
        self.n_restarts = opt_config.n_restarts

    def create_all_initialization_logits(
        self,
        n_products: int,
        returns_matrix: np.ndarray,
        scenario_mask: np.ndarray,
        risk_mask: np.ndarray,
        target_excess_return: float,
        expected_return_fm: np.ndarray
    ) -> torch.Tensor:
        """
        모든 초기화 전략을 배치로 생성

        Sobol 초기화 개선사항:
        - 단일 Sampler에서 N_SOBOL_STRATEGIES개 샘플을 연속 추출 (균등 분포 특성 활용)
        - 21,201 차원 제한 대응: 차원 축소 후 타일링으로 확장
        - [0,1] → 정규분포 역변환(ppf)으로 softmax logits에 적합한 스케일링

        Args:
            n_products: 상품 수
            returns_matrix: 수익률 행렬 [n_simulations, n_products]
            scenario_mask: 시나리오 마스크
            risk_mask: 위험자산 마스크
            target_excess_return: 목표 초과수익률
            expected_return_fm: Fama-MacBeth 기대수익률

        Returns:
            초기화 logits [n_restarts, n_products]
        """
        mean_returns = expected_return_fm
        std_returns = returns_matrix.std(axis=0)
        var_5 = np.percentile(returns_matrix, 5, axis=0)
        cvar_5 = np.array([returns_matrix[:, i][returns_matrix[:, i] <= var_5[i]].mean()
                           for i in range(n_products)])
        max_loss = returns_matrix.min(axis=0)

        all_logits = []

        # =========================================================================
        # Sobol Sampler 사전 생성 (단일 Sampler에서 연속 샘플링)
        # =========================================================================
        sobol_samples = None
        use_sobol = False

        if self.n_sobol > 0:
            try:
                # scipy.stats.qmc.Sobol은 최대 21,201 차원까지 지원
                SOBOL_MAX_DIM = 21201
                sobol_dim = min(n_products, SOBOL_MAX_DIM)

                # 단일 Sampler 생성 후 N_SOBOL_STRATEGIES개 샘플을 한번에 추출
                sobol_sampler = qmc.Sobol(d=sobol_dim, scramble=True, seed=42)
                sobol_samples = sobol_sampler.random(self.n_sobol)  # [n_sobol, sobol_dim]
                use_sobol = True

            except Exception as e:
                print(f"  ⚠️ Sobol 초기화 실패, 랜덤 초기화로 대체: {e}")
                use_sobol = False

        # =========================================================================
        # 전략별 초기화
        # =========================================================================
        for strategy_idx in range(self.n_restarts):
            if strategy_idx == 0:
                # 균등 배분
                init_logits = np.zeros(n_products)

            elif strategy_idx == 1:
                # 시나리오 집중
                init_logits = scenario_mask.astype(float) * 2.0

            elif strategy_idx == 2:
                # Return충족 + VaR (목표수익률 이상인 상품 중 VaR 좋은 것 선호)
                return_ok_mask = (mean_returns >= target_excess_return).astype(float)
                normalized_var = (var_5 - var_5.min()) / (var_5.max() - var_5.min() + 1e-6)
                init_logits = normalized_var * 2.0 * return_ok_mask + return_ok_mask * 1.0
                if return_ok_mask.sum() < 3:
                    init_logits = normalized_var * 3.0

            elif strategy_idx == 3:
                # 시나리오 + VaR
                normalized_var = (var_5 - var_5.min()) / (var_5.max() - var_5.min() + 1e-6)
                scenario_weight_init = scenario_mask.astype(float)
                init_logits = normalized_var * 2.0 * scenario_weight_init + scenario_weight_init * 1.0

            elif strategy_idx == 4:
                # 안전자산 집중
                safe_mask = ~risk_mask
                init_logits = safe_mask.astype(float) * 2.0

            elif strategy_idx == 5:
                # 저변동성 집중
                inv_std = 1.0 / (std_returns + 1e-6)
                normalized_inv_std = (inv_std - inv_std.min()) / (inv_std.max() - inv_std.min() + 1e-6)
                init_logits = normalized_inv_std * 3.0

            elif strategy_idx == 6:
                # CVaR상위 집중
                normalized_cvar = (cvar_5 - cvar_5.min()) / (cvar_5.max() - cvar_5.min() + 1e-6)
                init_logits = normalized_cvar * 3.0

            elif strategy_idx == 7:
                # 최대손실 회피
                normalized_max_loss = (max_loss - max_loss.min()) / (max_loss.max() - max_loss.min() + 1e-6)
                init_logits = normalized_max_loss * 3.0

            else:
                # =========================================================================
                # Sobol 기반 초기화
                # =========================================================================
                sobol_idx = strategy_idx - self.n_fixed

                if use_sobol and sobol_samples is not None:
                    # 사전 생성된 Sobol 샘플에서 해당 인덱스 추출
                    sobol_sample = sobol_samples[sobol_idx]  # [sobol_dim]

                    # 차원이 부족한 경우 타일링으로 확장
                    if len(sobol_sample) < n_products:
                        n_tiles = (n_products // len(sobol_sample)) + 1
                        sobol_sample = np.tile(sobol_sample, n_tiles)[:n_products]

                    # [0,1] 균등분포 → 정규분포 역변환 (ppf)
                    # 0과 1 근처 값은 무한대로 발산하므로 클리핑
                    sobol_sample_clipped = np.clip(sobol_sample, 0.001, 0.999)
                    init_logits = norm.ppf(sobol_sample_clipped) * 0.5  # 스케일 조정

                else:
                    # Fallback: 랜덤 초기화
                    np.random.seed(42 + sobol_idx)
                    init_logits = np.random.randn(n_products) * 0.5

            all_logits.append(init_logits)

        # [n_restarts, n_products] 형태로 스택 후 텐서 변환
        return torch.tensor(np.stack(all_logits), dtype=torch.float32, device=self.device)
