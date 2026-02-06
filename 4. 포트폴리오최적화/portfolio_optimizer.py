# -*- coding: utf-8 -*-
"""
포트폴리오 최적화 엔진
======================
포트폴리오 최적화

주요 기능:
- 16개 전략 동시 병렬 실행
- Functional Adam optimizer
- 전략별 독립적 LR 스케줄
- Early stopping
"""

import time
import numpy as np
import torch
from torch.func import vmap, grad
import warnings
warnings.filterwarnings('ignore')

from config import OptimizationConfig
from initialization import InitializationStrategy
from loss_functions import single_loss_fn, adam_step, cosine_lr_vectorized
from constraints import check_hard_constraints, get_constraint_violation_reasons
from utils import excess_to_total_return, get_strategy_name


# =============================================================================
# GPU 설정
# =============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()


# =============================================================================
# 포트폴리오 최적화 클래스
# =============================================================================

class PortfolioOptimizer:
    """
    병렬 포트폴리오 최적화 클래스

    주요 특징:
    - 16개 초기화 전략을 동시에 병렬 실행
    - Functional Adam optimizer
    - 전략별 독립적 Learning Rate Schedule
    - Early stopping
    """

    def __init__(self, opt_config: OptimizationConfig):
        """
        Args:
            opt_config: 최적화 설정 객체
        """
        self.opt_config = opt_config
        self.device = device
        self.initializer = InitializationStrategy(opt_config, device)

    def optimize(
        self,
        returns_matrix: np.ndarray,
        expected_return_fm: np.ndarray,
        target_excess_return: float,
        scenario_mask: np.ndarray,
        risk_mask: np.ndarray,
        tdf_mask: np.ndarray
    ) -> dict:
        """
        병렬 최적화

        Args:
            returns_matrix: 수익률 행렬 [n_simulations, n_products]
            expected_return_fm: Fama-MacBeth 기대수익률 [n_products]
            target_excess_return: 목표 초과수익률
            scenario_mask: 시나리오 마스크
            risk_mask: 위험자산 마스크
            tdf_mask: TDF 마스크

        Returns:
            최적화 결과 딕셔너리 (또는 실패시 {'status': 'failed'})
        """
        n_products = returns_matrix.shape[1]
        n_restarts = self.opt_config.n_restarts

        print(f"\n[최적화] 병렬 처리 ({n_products}개 상품, {n_restarts}개 전략 동시 실행)")
        print(f"  ★ 무위험 수익률: {self.opt_config.risk_free_rate*100:.3f}%")

        target_total_return = excess_to_total_return(target_excess_return, self.opt_config.risk_free_rate)
        print(f"  ★ 목표 총 수익률: ≥{target_total_return*100:.2f}%")

        # 텐서 준비 (GPU)
        returns_tensor = torch.tensor(returns_matrix, dtype=torch.float32, device=self.device)
        scenario_tensor = torch.tensor(scenario_mask, dtype=torch.float32, device=self.device)
        risk_tensor = torch.tensor(risk_mask, dtype=torch.float32, device=self.device)
        tdf_tensor = torch.tensor(tdf_mask, dtype=torch.float32, device=self.device)
        expected_return_fm_tensor = torch.tensor(expected_return_fm, dtype=torch.float32, device=self.device)

        # 하이퍼파라미터 텐서
        target_excess_tensor = torch.tensor(target_excess_return, dtype=torch.float32, device=self.device)
        scenario_min_tensor = torch.tensor(self.opt_config.scenario_min, dtype=torch.float32, device=self.device)
        risk_max_tensor = torch.tensor(self.opt_config.risk_asset_max, dtype=torch.float32, device=self.device)
        tdf_min_tensor = torch.tensor(self.opt_config.tdf_min, dtype=torch.float32, device=self.device)
        penalty_tensor = torch.tensor(self.opt_config.penalty_strength, dtype=torch.float32, device=self.device)
        temperature_tensor = torch.tensor(self.opt_config.temperature, dtype=torch.float32, device=self.device)

        # 모든 초기화 전략 생성 [n_restarts, n_products]
        all_logits = self.initializer.create_all_initialization_logits(
            n_products, returns_matrix,
            scenario_mask, risk_mask, target_excess_return,
            expected_return_fm
        )

        # Adam 상태 초기화 [n_restarts, n_products]
        m = torch.zeros_like(all_logits)  # 1차 모멘텀
        v = torch.zeros_like(all_logits)  # 2차 모멘텀

        # 전략별 독립적 step 카운터 초기화 [n_restarts]
        strategy_steps = torch.ones(n_restarts, dtype=torch.float32, device=self.device)

        # 배치 gradient 함수 생성
        batched_grad_fn = vmap(
            grad(single_loss_fn),
            in_dims=(0, None, None, None, None, None, None, None, None, None, None, None)
        )

        # 배치 loss 함수 생성 (모니터링용)
        batched_loss_fn = vmap(
            single_loss_fn,
            in_dims=(0, None, None, None, None, None, None, None, None, None, None, None)
        )

        # 학습 기록
        best_losses = torch.full((n_restarts,), float('inf'), device=self.device)
        best_logits = all_logits.clone()
        no_improve_counts = torch.zeros(n_restarts, device=self.device)
        active_mask = torch.ones(n_restarts, dtype=torch.bool, device=self.device)

        print(f"\n  학습 시작 (max {self.opt_config.n_epochs} epochs, patience={self.opt_config.patience})")
        print(f"  ★ 전략별 독립적 LR 스케줄 적용")

        # 학습 루프
        for epoch in range(self.opt_config.n_epochs):
            # 전략별 독립적 Learning rate 계산 [n_restarts]
            lr_vector = cosine_lr_vectorized(self.opt_config.learning_rate, strategy_steps, self.opt_config.n_epochs)

            # 배치 gradient 계산 (모든 전략 동시에)
            grads = batched_grad_fn(
                all_logits,
                returns_tensor,
                expected_return_fm_tensor,
                scenario_tensor,
                risk_tensor,
                tdf_tensor,
                target_excess_tensor,
                scenario_min_tensor,
                risk_max_tensor,
                tdf_min_tensor,
                penalty_tensor,
                temperature_tensor
            )

            # Functional Adam step (전략별 독립적 step과 LR 사용)
            all_logits, m, v = adam_step(
                all_logits, grads, m, v,
                steps=strategy_steps,
                lr=lr_vector
            )

            # Active한 전략만 step 증가 (early stopping된 전략은 step 동결)
            strategy_steps = strategy_steps + active_mask.float()

            # Early stopping 체크 (매 100 epoch)
            if (epoch + 1) % 100 == 0:
                with torch.no_grad():
                    current_losses = batched_loss_fn(
                        all_logits,
                        returns_tensor,
                        expected_return_fm_tensor,
                        scenario_tensor,
                        risk_tensor,
                        tdf_tensor,
                        target_excess_tensor,
                        scenario_min_tensor,
                        risk_max_tensor,
                        tdf_min_tensor,
                        penalty_tensor,
                        temperature_tensor
                    )

                    # 개선된 전략 업데이트
                    improved = current_losses < (best_losses - self.opt_config.tolerance)
                    best_losses = torch.where(improved, current_losses, best_losses)
                    best_logits = torch.where(improved.unsqueeze(1), all_logits, best_logits)

                    # No improve count 업데이트
                    no_improve_counts = torch.where(improved,
                                                    torch.zeros_like(no_improve_counts),
                                                    no_improve_counts + 100)

                    # Early stopping 체크
                    active_mask = no_improve_counts < self.opt_config.patience
                    n_active = active_mask.sum().item()

            # 진행 상황 출력
            if (epoch + 1) % 500 == 0:
                with torch.no_grad():
                    current_losses = batched_loss_fn(
                        all_logits,
                        returns_tensor,
                        expected_return_fm_tensor,
                        scenario_tensor,
                        risk_tensor,
                        tdf_tensor,
                        target_excess_tensor,
                        scenario_min_tensor,
                        risk_max_tensor,
                        tdf_min_tensor,
                        penalty_tensor,
                        temperature_tensor
                    )

                    min_loss = current_losses.min().item()
                    mean_loss = current_losses.mean().item()
                    n_active = active_mask.sum().item()

                    # 전략별 LR 통계 (min/max)
                    lr_min = lr_vector.min().item()
                    lr_max = lr_vector.max().item()

                    print(f"    Epoch {epoch+1}/{self.opt_config.n_epochs} | "
                          f"Loss (min/mean): {min_loss:.4f}/{mean_loss:.4f} | "
                          f"LR: {lr_min:.5f}~{lr_max:.5f} | "
                          f"Active: {n_active}/{n_restarts}")

            # 모든 전략이 수렴하면 종료
            if not active_mask.any():
                print(f"    모든 전략 수렴 at epoch {epoch+1}")
                break

        # 최종 결과 추출 및 검증
        print(f"\n  학습 완료. 후보 검증 중...")

        valid_candidates = []

        with torch.no_grad():
            # 배치로 softmax 계산 후 한번에 CPU로 전송
            all_weights = torch.softmax(best_logits, dim=-1).cpu().numpy()

            for restart in range(n_restarts):
                strategy_name = get_strategy_name(restart, self.opt_config.n_fixed_strategies)

                # 가중치 추출 및 정리
                final_weights = all_weights[restart].copy()
                n_active = np.sum(final_weights > 0.0001)  # 0.01% 초과 상품 수
                n_all = np.sum(final_weights > 0)  # 전체 상품 수 (0 초과)

                # 제약조건 검증
                is_valid, metrics = check_hard_constraints(
                    final_weights, returns_matrix, expected_return_fm, target_excess_return,
                    scenario_mask, risk_mask, tdf_mask, self.opt_config
                )

                if is_valid:
                    valid_candidates.append({
                        'weights': final_weights,
                        'var_95': metrics['var_95'],
                        'expected_return': metrics['expected_return'],
                        'expected_return_fm': metrics['expected_return_fm'],
                        'scenario_weight': metrics['scenario_weight'],
                        'risk_weight': metrics['risk_weight'],
                        'tdf_weight': metrics['tdf_weight'],
                        'n_active': n_active,
                        'n_all': n_all,
                        'best_strategy_idx': restart,
                        'best_strategy_name': strategy_name
                    })

                    print(f"    ✅ {strategy_name}: VaR={metrics['var_95']*100:.2f}%, "
                          f"TotalRet={metrics['expected_total_return']*100:.2f}%, "
                          f"TDF={metrics['tdf_weight']*100:.1f}%")
                else:
                    reasons = get_constraint_violation_reasons(metrics)
                    print(f"    ❌ {strategy_name}: {', '.join(reasons)}")

        if len(valid_candidates) == 0:
            print("\n⚠️ 모든 전략 실패 - 유효한 후보 없음")
            # 메모리 정리
            del returns_tensor, scenario_tensor, risk_tensor, tdf_tensor
            del expected_return_fm_tensor, all_logits, best_logits, m, v
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()
            return {'status': 'failed', 'reason': 'no_valid_candidates'}

        print(f"\n[최종 선택] 유효한 후보: {len(valid_candidates)}개")

        best_candidate = min(valid_candidates, key=lambda x: x['var_95'])

        expected_total_sim = excess_to_total_return(best_candidate['expected_return'], self.opt_config.risk_free_rate)
        expected_total_fm = excess_to_total_return(best_candidate['expected_return_fm'], self.opt_config.risk_free_rate)

        print(f"  → 채택: {best_candidate['best_strategy_name']}")
        print(f"   VaR 95%: {best_candidate['var_95']*100:.2f}%")
        print(f"   ★ 시뮬레이션 평균 수익률: {expected_total_sim*100:.2f}%")
        print(f"   ★ Fama-MacBeth r_hat 수익률: {expected_total_fm*100:.2f}%")
        print(f"   선택 상품 수: {best_candidate['n_active']}개")

        # 메모리 정리
        del returns_tensor, scenario_tensor, risk_tensor, tdf_tensor
        del expected_return_fm_tensor, all_logits, best_logits, m, v
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()

        return best_candidate
