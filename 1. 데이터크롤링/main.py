"""
데이터 크롤링 통합 실행 스크립트
- 6개 크롤러를 선택적으로 실행 가능
- 전체 실행 또는 개별 실행 지원
"""

import sys


def print_menu():
    """메뉴 출력"""
    print("\n" + "=" * 60)
    print("데이터 크롤링 통합 실행 스크립트")
    print("=" * 60)
    print("\n[가격 데이터 크롤러]")
    print("  1. ETF 가격 크롤러")
    print("  2. 리츠 가격 크롤러")
    print("  3. 펀드 가격 크롤러")
    print("\n[배당 데이터 크롤러]")
    print("  4. ETF 분배금 크롤러")
    print("  5. 리츠 배당 크롤러")
    print("  6. 펀드 분배금 크롤러")
    print("\n[일괄 실행]")
    print("  7. 가격 크롤러 전체 실행 (1, 2, 3)")
    print("  8. 배당 크롤러 전체 실행 (4, 5, 6)")
    print("  9. 전체 크롤러 실행 (1~6)")
    print("\n  0. 종료")
    print("=" * 60)


def run_etf_price():
    """ETF 가격 크롤러 실행"""
    print("\n[ETF 가격 크롤러 실행]")
    import etf_price_crawler
    etf_price_crawler.run()


def run_reits_price():
    """리츠 가격 크롤러 실행"""
    print("\n[리츠 가격 크롤러 실행]")
    import reits_price_crawler
    reits_price_crawler.run()


def run_fund_price():
    """펀드 가격 크롤러 실행"""
    print("\n[펀드 가격 크롤러 실행]")
    import fund_price_crawler
    fund_price_crawler.run()


def run_etf_dividend():
    """ETF 분배금 크롤러 실행"""
    print("\n[ETF 분배금 크롤러 실행]")
    import etf_dividend_crawler
    etf_dividend_crawler.run()


def run_reits_dividend():
    """리츠 배당 크롤러 실행"""
    print("\n[리츠 배당 크롤러 실행]")
    import reits_dividend_crawler
    reits_dividend_crawler.run()


def run_fund_dividend():
    """펀드 분배금 크롤러 실행"""
    print("\n[펀드 분배금 크롤러 실행]")
    import fund_dividend_crawler
    fund_dividend_crawler.run()


def run_all_price():
    """가격 크롤러 전체 실행"""
    print("\n[가격 크롤러 전체 실행]")
    run_etf_price()
    run_reits_price()
    run_fund_price()


def run_all_dividend():
    """배당 크롤러 전체 실행"""
    print("\n[배당 크롤러 전체 실행]")
    run_etf_dividend()
    run_reits_dividend()
    run_fund_dividend()


def run_all():
    """전체 크롤러 실행"""
    print("\n[전체 크롤러 실행]")
    run_all_price()
    run_all_dividend()


def execute_choice(choice):
    """선택된 작업 실행"""
    if choice == '1':
        run_etf_price()
    elif choice == '2':
        run_reits_price()
    elif choice == '3':
        run_fund_price()
    elif choice == '4':
        run_etf_dividend()
    elif choice == '5':
        run_reits_dividend()
    elif choice == '6':
        run_fund_dividend()
    elif choice == '7':
        run_all_price()
    elif choice == '8':
        run_all_dividend()
    elif choice == '9':
        run_all()
    else:
        print("\n잘못된 입력입니다.")
        return False
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
