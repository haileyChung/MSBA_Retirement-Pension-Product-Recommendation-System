# -*- coding: utf-8 -*-
"""
개인맞춤 리포트 생성 메인 실행 파일

Multi-AI Agent 시스템을 사용하여 개인맞춤 투자 리포트를 생성합니다.
"""

import asyncio
import sys
from pathlib import Path

from config import load_config
from orchestrator import ReportOrchestrator


# ==================================================================================
# 메뉴 시스템
# ==================================================================================

def print_header():
    """헤더 출력"""
    print("\n" + "=" * 80)
    print("개인맞춤 리포트 생성 시스템")
    print("Multi-AI Agent Architecture (Orchestrator + 3 Agents)")
    print("=" * 80 + "\n")


def print_menu():
    """메뉴 출력"""
    print("\n" + "-" * 80)
    print("[메뉴]")
    print("  1. OCR + NER 파이프라인 실행 (DB 적재)")
    print("  2. 개인맞춤 리포트 생성 (단일 조합)")
    print("  3. 설정 확인")
    print("  0. 종료")
    print("-" * 80)


def show_config(config):
    """설정 확인"""
    print("\n[설정 정보]")
    print(f"  - 프로젝트 루트: {config.path.project_root}")
    print(f"  - 포트폴리오 DB: {config.path.portfolio_db_path}")
    print(f"  - 상품 마스터: {config.path.product_master_path}")
    print(f"  - PDF 입력 폴더: {config.path.input_pdf_folder}")
    print(f"  - OCR 출력 폴더: {config.path.output_ocr_folder}")
    print(f"  - NER 출력 폴더: {config.path.output_ner_folder}")
    print(f"  - 인사이트 DB: {config.path.output_insights_db}")
    print(f"  - 리포트 출력 폴더: {config.path.output_reports_folder}")
    print()
    print(f"  - OCR 모델: {config.ocr.model_name}")
    print(f"  - NER 모델: {config.ner.model_name}")
    print(f"  - 리포트 모델: {config.report.model_name}")
    print()
    print(f"  - 네이버 API: {config.retrieval.naver_client_id[:4]}... (설정됨)"
          if config.retrieval.naver_client_id else "  - 네이버 API: 미설정")


# ==================================================================================
# 메인 실행 함수
# ==================================================================================

def run_ocr_ner_pipeline(orchestrator: ReportOrchestrator):
    """OCR + NER 파이프라인 실행"""
    print("\n[실행] OCR + NER 파이프라인")
    success = orchestrator.run_ocr_ner_pipeline()

    if success:
        print("\n[성공] OCR + NER 파이프라인 완료")
    else:
        print("\n[실패] OCR + NER 파이프라인 오류 발생")


async def generate_single_report(orchestrator: ReportOrchestrator):
    """단일 조합 리포트 생성"""
    print("\n[입력] 포트폴리오 조건 입력")

    # 지역 입력
    print("\n사용 가능한 지역:")
    regions = orchestrator.config.ner.region_choices
    for i, r in enumerate(regions, 1):
        print(f"  {i}. {r}")

    region_idx = int(input("지역 선택 (번호): ")) - 1
    region = regions[region_idx]

    # 테마 입력
    print("\n사용 가능한 테마:")
    themes = orchestrator.config.ner.theme_choices
    for i, t in enumerate(themes, 1):
        print(f"  {i}. {t}")

    theme_idx = int(input("테마 선택 (번호): ")) - 1
    theme = themes[theme_idx]

    # 목표 수익률 입력
    target_return_pct = float(input("\n목표 수익률 (%, 예: 7): "))
    target_return = target_return_pct / 100

    # 은퇴연도 입력
    retire_year = int(input("은퇴연도 (예: 2045): "))

    # 리포트 생성
    report = await orchestrator.generate_personalized_report(
        region=region,
        theme=theme,
        target_return=target_return,
        retire_year=retire_year,
        save_to_file=True
    )

    if report:
        print("\n[성공] 리포트 생성 완료")
        print(f"\n[요약]")
        print(report['summary'][:200] + "...")
    else:
        print("\n[실패] 리포트 생성 실패")


async def main_async():
    """비동기 메인 함수"""
    print_header()

    # 설정 로드
    try:
        config = load_config()
    except ValueError as e:
        print(f"[오류] 설정 로드 실패: {e}")
        return

    # 오케스트레이터 초기화
    try:
        orchestrator = ReportOrchestrator(config)
    except Exception as e:
        print(f"[오류] 오케스트레이터 초기화 실패: {e}")
        import traceback
        traceback.print_exc()
        return

    # 메뉴 루프
    while True:
        print_menu()
        choice = input("\n선택: ").strip()

        if choice == "1":
            run_ocr_ner_pipeline(orchestrator)

        elif choice == "2":
            await generate_single_report(orchestrator)

        elif choice == "3":
            show_config(config)

        elif choice == "0":
            print("\n프로그램을 종료합니다.")
            break

        else:
            print("\n[오류] 잘못된 선택입니다.")


def main():
    """메인 함수"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\n\n프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"\n[오류] 예상치 못한 오류 발생: {e}")
        import traceback
        traceback.print_exc()


# ==================================================================================
# Command-line 실행
# ==================================================================================

def run_cli():
    """Command-line 인터페이스"""
    if len(sys.argv) < 2:
        # 인자 없으면 메뉴 모드
        main()
        return

    command = sys.argv[1]

    if command == "ocr-ner":
        # OCR + NER 실행
        config = load_config()
        orchestrator = ReportOrchestrator(config)
        orchestrator.run_ocr_ner_pipeline()

    elif command == "report":
        # 리포트 생성 (인자: region theme target_return retire_year)
        if len(sys.argv) < 6:
            print("사용법: python main.py report <지역> <테마> <목표수익률> <은퇴연도>")
            print("예시: python main.py report 한국 반도체 0.08 2045")
            return

        region = sys.argv[2]
        theme = sys.argv[3]
        target_return = float(sys.argv[4])
        retire_year = int(sys.argv[5])

        config = load_config()
        orchestrator = ReportOrchestrator(config)

        async def run():
            await orchestrator.generate_personalized_report(
                region, theme, target_return, retire_year
            )

        asyncio.run(run())

    else:
        print(f"[오류] 알 수 없는 명령: {command}")
        print("사용 가능한 명령: ocr-ner, report")


# ==================================================================================
# 실행
# ==================================================================================

if __name__ == "__main__":
    run_cli()
