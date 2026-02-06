# -*- coding: utf-8 -*-
"""
포트폴리오 최적화 메인 실행 파일
================================

실행 모드:
- 단일 조합 최적화
- Portfolio Options Combinations 포트폴리오옵션조합
"""

import sys
import time
from itertools import product

from config import PathConfig, OptimizationConfig, create_configs
from data_loader import PortfolioDataLoader
from portfolio_optimizer import PortfolioOptimizer
from database import create_database, save_to_database, check_existing_result
from utils import get_target_excess_return, excess_to_total_return


# =============================================================================
# 단일 조합 최적화 함수
# =============================================================================

def run_single_optimization(
    preferred_regions: list,
    preferred_themes: list,
    target_return: float,
    target_retirement_year: int
) -> dict:
    """
    단일 조합 최적화 실행

    Args:
        preferred_regions: 선호 지역 리스트
        preferred_themes: 선호 테마 리스트
        target_return: 목표 총 수익률
        target_retirement_year: 은퇴 예정 연도

    Returns:
        최적화 결과 딕셔너리
    """
    path_cfg, opt_cfg = create_configs()

    print("=" * 80)
    print("포트폴리오 최적화")
    print("  - 목적함수: VaR 95% 최소화")
    print("  - 기대수익률: Fama-MacBeth r_hat (초과수익률)")
    print(f"  - ★ 무위험 수익률: {opt_cfg.risk_free_rate*100:.3f}% (KOFR 11/13)")
    print(f"  - ★ 병렬 처리: {opt_cfg.n_restarts}개 전략 동시 실행")
    print(f"  - 제약조건: 시나리오 ≥{opt_cfg.scenario_min*100:.0f}%, "
          f"위험상품 ≤{opt_cfg.risk_asset_max*100:.0f}%, TDF ≥{opt_cfg.tdf_min*100:.0f}%")
    print(f"  - 은퇴 예정 연도: {target_retirement_year}년 "
          f"(TDF {target_retirement_year} 이하 상품만 포함)")
    print("=" * 80)

    target_excess_return = get_target_excess_return(target_return, opt_cfg.risk_free_rate)

    print(f"\n[설정]")
    print(f"  선호 지역: {', '.join(preferred_regions)}")
    print(f"  선호 테마: {', '.join(preferred_themes)}")
    print(f"  ★ 고객 희망 총 수익률: {target_return*100:.0f}%")
    print(f"    → 초과수익률 목표: ≥{target_excess_return*100:.2f}%")

    # DB 생성
    create_database(path_cfg.database_path)

    # 중복 체크
    existing_id = check_existing_result(
        path_cfg.database_path,
        preferred_regions, preferred_themes,
        target_return, target_retirement_year
    )
    if existing_id:
        print(f"\n⏭️  동일 조건 이미 존재 (portfolio_id: {existing_id})")
        return {'status': 'skipped', 'reason': 'duplicate'}

    # 데이터 로드
    data_loader = PortfolioDataLoader(path_cfg, opt_cfg)
    (df_products, simulation_data, available_codes, returns_matrix,
     expected_return_fm, scenario_mask, risk_mask, tdf_mask) = data_loader.load_all(
        preferred_regions, preferred_themes, target_retirement_year
    )

    # 최적화 실행
    print("\n[최적화 실행]")
    print("-" * 60)

    optimizer = PortfolioOptimizer(opt_cfg)
    start_time = time.time()
    result = optimizer.optimize(
        returns_matrix, expected_return_fm, target_excess_return,
        scenario_mask, risk_mask, tdf_mask
    )
    elapsed_time = time.time() - start_time

    # DB 저장
    print("\n[DB 저장]")
    print("-" * 60)

    save_to_database(
        path_cfg.database_path, result, df_products, available_codes, elapsed_time,
        preferred_regions, preferred_themes, target_return, target_retirement_year,
        opt_cfg, scenario_mask, tdf_mask, returns_matrix, expected_return_fm
    )

    # 결과 출력
    if result and result.get('status') != 'failed':
        expected_total_sim = excess_to_total_return(result['expected_return'], opt_cfg.risk_free_rate)
        expected_total_fm = excess_to_total_return(result['expected_return_fm'], opt_cfg.risk_free_rate)

        print("\n" + "=" * 80)
        print("최적화 결과")
        print("=" * 80)

        print(f"\n🎯 채택된 전략: {result['best_strategy_name']}")

        print(f"\n📊 성과 지표:")
        print(f"  VaR 95%: {result['var_95']*100:.2f}%")
        print(f"\n  ★ 시뮬레이션 평균 수익률:")
        print(f"    - 총 수익률: {expected_total_sim*100:.2f}%")
        print(f"    - 초과수익률: {result['expected_return']*100:.2f}%")
        print(f"\n  ★ Fama-MacBeth r_hat 수익률:")
        print(f"    - 총 수익률: {expected_total_fm*100:.2f}%")
        print(f"    - 초과수익률: {result['expected_return_fm']*100:.2f}%")

        print(f"\n📋 제약조건:")
        print(f"  시나리오 비중: {result['scenario_weight']*100:.1f}% "
              f"(목표: ≥{opt_cfg.scenario_min*100:.0f}%)")
        print(f"  위험상품 비중: {result['risk_weight']*100:.1f}% "
              f"(목표: ≤{opt_cfg.risk_asset_max*100:.0f}%)")
        print(f"  TDF 비중: {result['tdf_weight']*100:.1f}% "
              f"(목표: ≥{opt_cfg.tdf_min*100:.0f}%, 은퇴연도 {target_retirement_year} 이하)")
        print(f"  활성 상품 수: {result['n_active']}개")
        print(f"\n  ⏱️ 소요 시간: {elapsed_time:.1f}초")

    return result


# =============================================================================
# Portfolio Options Combinations 포트폴리오옵션조합
# =============================================================================

def run_grid_search():
    """모든 조합에 대해 Portfolio Options Combinations 실행"""
    path_cfg, opt_cfg = create_configs()

    all_combinations = list(product(
        opt_cfg.region_options,
        opt_cfg.theme_options,
        opt_cfg.target_return_options,
        opt_cfg.target_retirement_year_options
    ))

    total_combinations = len(all_combinations)
    print("=" * 80)
    print(f"  - ★ {opt_cfg.n_restarts}개 전략 동시 병렬 처리")
    print(f"  - ★ 무위험 수익률: {opt_cfg.risk_free_rate*100:.3f}%")
    print("=" * 80)
    print(f"\n총 조합 수: {total_combinations}개")
    print(f"  - 지역 옵션: {opt_cfg.region_options}")
    print(f"  - 테마 옵션 수: {len(opt_cfg.theme_options)}개")
    print(f"  - 목표 수익률 옵션 수: {len(opt_cfg.target_return_options)}개")
    print(f"  - 은퇴연도 옵션 수: {len(opt_cfg.target_retirement_year_options)}개")

    create_database(path_cfg.database_path)

    success_count = 0
    skip_count = 0
    fail_count = 0
    results_summary = []

    total_start_time = time.time()

    for idx, (region, theme, target_return, retirement_year) in enumerate(all_combinations):
        print("\n" + "=" * 80)
        print(f"[조합 {idx+1}/{total_combinations}]")

        target_excess = get_target_excess_return(target_return, opt_cfg.risk_free_rate)

        print(f"  지역: {region}, 테마: {theme}")
        print(f"  ★ 목표 총 수익률: {target_return*100:.0f}% → 초과수익률 목표: {target_excess*100:.2f}%")
        print(f"  ★ 은퇴 예정 연도: {retirement_year}년")
        print("=" * 80)

        try:
            result = run_single_optimization([region], [theme], target_return, retirement_year)
            if result is None:
                fail_count += 1
                results_summary.append({
                    'region': region, 'theme': theme, 'target_return': target_return,
                    'retirement_year': retirement_year, 'status': 'failed', 'error': 'unexpected None'
                })
            elif result.get('status') == 'skipped':
                skip_count += 1
                results_summary.append({
                    'region': region, 'theme': theme, 'target_return': target_return,
                    'retirement_year': retirement_year, 'status': 'skipped'
                })
            elif result.get('status') == 'failed':
                fail_count += 1
                results_summary.append({
                    'region': region, 'theme': theme, 'target_return': target_return,
                    'retirement_year': retirement_year, 'status': 'failed',
                    'error': result.get('reason', 'optimization failed')
                })
            else:
                success_count += 1
                results_summary.append({
                    'region': region, 'theme': theme, 'target_return': target_return,
                    'retirement_year': retirement_year, 'status': 'success',
                    'expected_total_return': excess_to_total_return(result['expected_return'], opt_cfg.risk_free_rate),
                    'expected_excess_return': result['expected_return'],
                    'var_95': result['var_95']
                })
        except Exception as e:
            fail_count += 1
            print(f"❌ 오류 발생: {e}")
            import traceback
            traceback.print_exc()
            results_summary.append({
                'region': region, 'theme': theme, 'target_return': target_return,
                'retirement_year': retirement_year, 'status': 'failed', 'error': str(e)
            })

    total_elapsed = time.time() - total_start_time

    print("\n" + "=" * 80)
    print("Portfolio Options Combinations 완료 - 최종 요약")
    print("=" * 80)
    print(f"  총 조합 수: {total_combinations}개")
    print(f"  ✅ 성공: {success_count}개")
    print(f"  ⏭️  건너뜀 (중복): {skip_count}개")
    print(f"  ❌ 실패: {fail_count}개")
    print(f"  ⏱️ 총 소요 시간: {total_elapsed:.1f}초")
    if success_count > 0:
        print(f"  ⏱️ 평균 소요 시간: {total_elapsed/success_count:.1f}초/조합")

    return results_summary


# =============================================================================
# 메뉴 출력 및 메인 함수
# =============================================================================

def print_menu():
    """메뉴 출력"""
    print("\n" + "=" * 60)
    print("포트폴리오 최적화 - 메뉴")
    print("=" * 60)
    print("1. 단일 조합 최적화")
    print("2. Portfolio Options Combinations 포트폴리오옵션조합")
    print("3. 설정 확인")
    print("0. 종료")
    print("=" * 60)


def main():
    """메인 함수"""
    # Command-line 인자 확인
    if len(sys.argv) > 1:
        if sys.argv[1] == "grid":
            run_grid_search()
            return
        elif sys.argv[1] == "single":
            # 예시: python main.py single 한국 반도체 0.08 2045
            if len(sys.argv) >= 6:
                region = sys.argv[2]
                theme = sys.argv[3]
                target_return = float(sys.argv[4])
                retirement_year = int(sys.argv[5])
                run_single_optimization([region], [theme], target_return, retirement_year)
                return

    # Interactive 모드
    while True:
        print_menu()
        choice = input("선택: ").strip()

        if choice == "1":
            # 단일 조합 최적화 (예시 값 사용)
            run_single_optimization(
                preferred_regions=['한국'],
                preferred_themes=['반도체'],
                target_return=0.08,
                target_retirement_year=2045
            )

        elif choice == "2":
            # Grid Search
            run_grid_search()

        elif choice == "3":
            # 설정 확인
            from config import create_configs
            path_cfg, opt_cfg = create_configs()
            print("\n[경로 설정]")
            print(f"  DB 경로: {path_cfg.database_path}")
            print(f"  Output 폴더: {path_cfg.output_dir}")
            print("\n[최적화 설정]")
            print(f"  무위험 수익률: {opt_cfg.risk_free_rate*100:.3f}%")
            print(f"  초기화 전략 수: {opt_cfg.n_restarts}개")
            print(f"  학습 에포크: {opt_cfg.n_epochs}회")
            print(f"  초기 학습률: {opt_cfg.learning_rate}")

        elif choice == "0":
            print("종료합니다.")
            break

        else:
            print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()
