# -*- coding: utf-8 -*-
"""
손실 함수 모듈
==============
포트폴리오 최적화를 위한 손실 함수 정의

주요 기능:
- Soft Quantile (미분 가능한 VaR 근사)
- Functional Loss Function
- Functional Adam Optimizer
- Cosine Learning Rate Schedule
"""

import numpy as np
import torch


# =============================================================================
# Soft Quantile 함수
# =============================================================================

def soft_quantile_functional(x: torch.Tensor, q: float = 0.05, temperature: float = 1.0) -> torch.Tensor:
    """
    Functional Soft Quantile

    단일 포트폴리오의 수익률 벡터에서 soft VaR 계산

    Args:
        x: 수익률 벡터 [n_simulations]
        q: 분위수 (VaR 95%의 경우 0.05)
        temperature: Softmax temperature (클수록 더 smooth)

    Returns:
        Soft quantile 값
    """
    n = x.shape[0]
    sorted_x, _ = torch.sort(x)
    target_idx = q * (n - 1)

    # Window 기반 계산 (메모리 효율)
    window_size = min(1000, n // 10)
    start_idx = max(0, int(target_idx) - window_size // 2)
    end_idx = min(n, start_idx + window_size)

    indices = torch.arange(start_idx, end_idx, dtype=torch.float32, device=x.device)
    weights = torch.softmax(-(indices - target_idx)**2 / temperature, dim=0)

    return (sorted_x[start_idx:end_idx] * weights).sum()


# =============================================================================
# 단일 전략 Loss Function
# =============================================================================

def single_loss_fn(
    logits: torch.Tensor,
    returns_matrix: torch.Tensor,
    expected_return_fm: torch.Tensor,
    scenario_mask: torch.Tensor,
    risk_mask: torch.Tensor,
    tdf_mask: torch.Tensor,
    target_excess: torch.Tensor,
    scenario_min: torch.Tensor,
    risk_max: torch.Tensor,
    tdf_min: torch.Tensor,
    penalty_strength: torch.Tensor,
    temperature: torch.Tensor
) -> torch.Tensor:
    """
    단일 전략에 대한 Loss 계산

    Args:
        logits: [n_products] - 단일 전략의 logits
        returns_matrix: [n_simulations, n_products]
        expected_return_fm: [n_products]
        scenario_mask: [n_products]
        risk_mask: [n_products]
        tdf_mask: [n_products] - 은퇴연도 이하 TDF 상품 마스크
        target_excess: 스칼라 - 목표 초과수익률
        scenario_min: 스칼라 - 시나리오 최소 비중
        risk_max: 스칼라 - 위험자산 최대 비중
        tdf_min: 스칼라 - TDF 최소 비중
        penalty_strength: 스칼라 - 페널티 강도
        temperature: 스칼라 - Soft quantile temperature

    Returns:
        loss: 스칼라
    """
    # Softmax로 가중치 계산
    weights = torch.softmax(logits, dim=-1)  # [n_products]

    # 포트폴리오 수익률 계산
    port_ret = returns_matrix @ weights  # [n_simulations]

    # Soft VaR 계산
    soft_var = soft_quantile_functional(port_ret, q=0.05, temperature=temperature)
    loss_var = -soft_var  # 최소화 → 음수

    # 기대수익률 (Fama-MacBeth r_hat 사용)
    exp_ret_fm = (weights * expected_return_fm).sum()

    # 제약조건 비중 계산
    scen_w = (weights * scenario_mask).sum()
    risk_w = (weights * risk_mask).sum()
    tdf_w = (weights * tdf_mask).sum()

    # 제약조건 위반 페널티
    viol_ret = torch.relu(target_excess - exp_ret_fm)
    viol_scen_min = torch.relu(scenario_min - scen_w)
    viol_risk = torch.relu(risk_w - risk_max)
    viol_tdf = torch.relu(tdf_min - tdf_w)

    total_violation = viol_ret + viol_scen_min + viol_risk + viol_tdf

    # 최종 Loss
    loss = loss_var + (total_violation * penalty_strength)

    return loss


# =============================================================================
# Functional Adam Optimizer
# =============================================================================

def adam_step(
    params: torch.Tensor,
    grads: torch.Tensor,
    m: torch.Tensor,
    v: torch.Tensor,
    steps: torch.Tensor,
    lr: torch.Tensor,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8
) -> tuple:
    """
    Functional Adam optimizer step - 전략별 독립적 LR 지원

    모든 입력이 배치 차원을 가질 수 있음

    Args:
        params: 현재 파라미터 [batch, n_products]
        grads: gradient [batch, n_products]
        m: 1차 모멘텀 [batch, n_products]
        v: 2차 모멘텀 [batch, n_products]
        steps: 전략별 현재 스텝 [batch] - 각 전략의 독립적 step
        lr: 전략별 learning rate [batch] - 각 전략의 독립적 LR
        beta1: 1차 모멘텀 decay 계수
        beta2: 2차 모멘텀 decay 계수
        eps: 수치 안정성을 위한 작은 값

    Returns:
        (new_params, new_m, new_v): 업데이트된 파라미터 및 모멘텀
    """
    # Momentum 업데이트
    new_m = beta1 * m + (1 - beta1) * grads
    new_v = beta2 * v + (1 - beta2) * (grads ** 2)

    # Bias correction (전략별 독립적 step 사용)
    # steps: [batch] → [batch, 1]로 확장하여 broadcasting
    steps_expanded = steps.unsqueeze(1)  # [batch, 1]
    m_hat = new_m / (1 - beta1 ** steps_expanded)
    v_hat = new_v / (1 - beta2 ** steps_expanded)

    # Parameter 업데이트 (전략별 독립적 LR 사용)
    # lr: [batch] → [batch, 1]로 확장하여 broadcasting
    lr_expanded = lr.unsqueeze(1)  # [batch, 1]
    new_params = params - lr_expanded * m_hat / (torch.sqrt(v_hat) + eps)

    return new_params, new_m, new_v


# =============================================================================
# Learning Rate Schedule
# =============================================================================

def cosine_lr_vectorized(
    initial_lr: float,
    steps: torch.Tensor,
    total_steps: int
) -> torch.Tensor:
    """
    Vectorized Cosine annealing learning rate schedule

    Args:
        initial_lr: 초기 learning rate (스칼라)
        steps: 전략별 현재 step [batch] 텐서
        total_steps: 총 epoch 수 (스칼라)

    Returns:
        lr_vector: 전략별 learning rate [batch] 텐서
    """
    return initial_lr * (1 + torch.cos(np.pi * steps / total_steps)) / 2
