# -*- coding: utf-8 -*-
"""
데이터 전처리 파이프라인
========================
ETF/펀드/REITs 데이터를 전처리하여 분석용 데이터셋을 생성합니다.

실행 단계:
    1. 수익률 계산: 가격/배당 데이터 → 초과수익률
    2. 상품명 생성: 초과수익률 → 상품명.xlsx
    3. 상품 분류: LLM 기반 국가/테마 분류 (선택)
    4. 분석기간 설정: 데이터 포인트 분석 (선택)
"""

import sys


def print_menu():
    """메뉴 출력"""
    print("\n" + "=" * 60)
    print("데이터 전처리 파이프라인")
    print("=" * 60)
    print("\n[필수 단계]")
    print("  1. 수익률 계산 (가격/배당 → 초과수익률)")
    print("  2. 상품명 생성 (초과수익률 → 상품명.xlsx)")
    print("\n[선택 단계]")
    print("  3. 상품 분류 LLM (국가/테마 분류)")
    print("  4. 분석기간 설정 (데이터 포인트 분석)")
    print("\n[일괄 실행]")
    print("  5. 필수 단계 실행 (1, 2)")
    print("  6. 전체 단계 실행 (1 ~ 4)")
    print("\n  0. 종료")
    print("=" * 60)


def run_step1():
    """1단계: 수익률 계산"""
    print("\n" + "=" * 70)
    print("  [1단계] 수익률 계산")
    print("=" * 70)

    from importlib import import_module
    module = import_module("1_수익률계산")

    if hasattr(module, "run_return_calculation"):
        module.run_return_calculation()
    elif hasattr(module, "main"):
        module.main()
    else:
        # ReturnCalculationPipeline 직접 실행
        from config import ReturnConfig
        pipeline = module.ReturnCalculationPipeline(ReturnConfig())
        pipeline.run()


def run_step2():
    """2단계: 상품명 생성"""
    print("\n" + "=" * 70)
    print("  [2단계] 상품명 생성")
    print("=" * 70)

    from importlib import import_module
    module = import_module("2_상품명생성")

    if hasattr(module, "run_product_name_generation"):
        module.run_product_name_generation()
    else:
        module.main()


def run_step3():
    """3단계: 상품 분류 LLM (선택)"""
    print("\n" + "=" * 70)
    print("  [3단계] 상품 분류 (LLM)")
    print("=" * 70)

    from importlib import import_module
    module = import_module("3_상품분류_LLM")
    from config import LLMConfig

    classifier = module.ProductClassifier(LLMConfig())
    classifier.run()


def run_step4():
    """4단계: 분석기간 설정 (선택)"""
    print("\n" + "=" * 70)
    print("  [4단계] 분석기간 설정")
    print("=" * 70)

    from importlib import import_module
    module = import_module("4_분석기간설정")

    if hasattr(module, "run_analysis_period_check"):
        module.run_analysis_period_check()
    else:
        checker = module.AnalysisPeriodChecker()
        checker.run()


def run_required():
    """필수 단계 실행 (1-2)"""
    print("\n[필수 단계 실행]")
    run_step1()
    run_step2()


def run_all():
    """전체 단계 실행 (1-4)"""
    print("\n[전체 단계 실행]")
    run_step1()
    run_step2()
    run_step3()
    run_step4()


def execute_choice(choice):
    """선택된 작업 실행"""
    if choice == '1':
        run_step1()
    elif choice == '2':
        run_step2()
    elif choice == '3':
        run_step3()
    elif choice == '4':
        run_step4()
    elif choice == '5':
        run_required()
    elif choice == '6':
        run_all()
    else:
        print("\n잘못된 입력입니다.")
        return False

    print("\n" + "#" * 70)
    print("#  작업 완료")
    print("#" * 70)
    return True


def main():
    """메인 함수"""
    # 커맨드라인 인자로 실행
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        execute_choice(choice)
    else:
        # 대화형 모드
        while True:
            print_menu()
            choice = input("\n실행할 작업 번호를 입력하세요: ").strip()

            if choice == '0':
                print("\n프로그램을 종료합니다.")
                break

            execute_choice(choice)
            input("\n계속하려면 Enter를 누르세요...")


if __name__ == "__main__":
    main()
