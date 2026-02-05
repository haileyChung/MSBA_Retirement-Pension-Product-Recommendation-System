# -*- coding: utf-8 -*-
"""
분석기간 설정 및 데이터 포인트 분석
==================================
각 기간별로 충분한 데이터를 보유한 상품 수를 분석합니다.

기능:
    1. 전체 Market Factor 분석용 데이터 포인트 계산
    2. 카테고리별(지역, 주식채권구분, 테마) 데이터 포인트 분석
    3. 코드 매칭 검증 (초과수익률 vs 상품정보)

입력:
    - 초과수익률 데이터 (1_수익률계산.py 결과)
    - 상품 정보 파일 (상품명.xlsx)

출력:
    - DataPoints_for_MarketFactor.csv/png
    - DataPoints_by_{카테고리}.csv/png
    - 코드 매칭 검증 결과
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from typing import Dict

# 설정 파일 임포트
from config import AnalysisPeriodConfig


class AnalysisPeriodChecker:
    """분석기간별 데이터 포인트 분석 클래스"""

    def __init__(self, config: AnalysisPeriodConfig = None):
        self.config = config or AnalysisPeriodConfig()
        self.df = None
        self.df_product = None

        # 한글 폰트 설정
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False

        # 출력 폴더 생성
        os.makedirs(self.config.output_dir, exist_ok=True)

    def load_data(self) -> None:
        """데이터 로드"""
        print("=" * 70)
        print("[1] 데이터 로드")
        print("=" * 70)

        # 초과수익률 데이터 로드
        print(f"  초과수익률: {self.config.path_excess_return}")
        self.df = pd.read_csv(self.config.path_excess_return, dtype=str)
        print(f"  -> {len(self.df):,}행 로드")

        # 상품 정보 로드
        print(f"  상품정보: {self.config.path_product_info}")
        self.df_product = pd.read_excel(self.config.path_product_info)
        print(f"  -> {len(self.df_product):,}개 상품 로드")

        # 전처리
        self.df['기준일자'] = pd.to_datetime(self.df['기준일자'])
        self.df = self.df.sort_values('기준일자', ascending=False)

        # 코드 정규화
        self.df['코드'] = self.df['코드'].astype(str).str.strip().str.upper()
        self.df_product['상품 코드'] = self.df_product['상품 코드'].astype(str).str.strip().str.upper()

    def verify_code_matching(self) -> Dict[str, set]:
        """코드 매칭 검증"""
        print("\n" + "=" * 70)
        print("[2] 코드 매칭 검증")
        print("=" * 70)

        codes_in_csv = set(self.df['코드'].unique())
        codes_in_excel = set(self.df_product['상품 코드'].dropna().unique())

        codes_only_in_excel = codes_in_excel - codes_in_csv
        codes_only_in_csv = codes_in_csv - codes_in_excel
        codes_matched = codes_in_csv & codes_in_excel

        print(f"\n  [초과수익률 파일]")
        print(f"    - 총 행 수: {len(self.df):,}")
        print(f"    - 유니크 코드 수: {len(codes_in_csv)}")

        print(f"\n  [상품정보 파일]")
        print(f"    - 총 상품 수: {len(self.df_product)}")
        print(f"    - 유니크 코드 수: {len(codes_in_excel)}")

        print(f"\n  [매칭 결과]")
        print(f"    - 매칭된 코드: {len(codes_matched)}개")
        print(f"    - 상품정보에만 존재: {len(codes_only_in_excel)}개")
        print(f"    - 초과수익률에만 존재: {len(codes_only_in_csv)}개")

        # 매칭 안 되는 코드 저장
        if len(codes_only_in_excel) > 0:
            df_only_excel = self.df_product[
                self.df_product['상품 코드'].isin(codes_only_in_excel)
            ][['상품 코드', '상품명', '지역', '주식채권구분', '테마']]
            path = os.path.join(self.config.output_dir, '코드_상품정보에만존재.csv')
            df_only_excel.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"\n    -> 저장: 코드_상품정보에만존재.csv")

        if len(codes_only_in_csv) > 0:
            code_counts = self.df[self.df['코드'].isin(codes_only_in_csv)].groupby('코드').size()
            df_only_csv = code_counts.reset_index(name='데이터_개수')
            path = os.path.join(self.config.output_dir, '코드_초과수익률에만존재.csv')
            df_only_csv.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"    -> 저장: 코드_초과수익률에만존재.csv")

        return {
            'matched': codes_matched,
            'only_excel': codes_only_in_excel,
            'only_csv': codes_only_in_csv
        }

    def analyze_market_factor(self) -> pd.DataFrame:
        """전체 Market Factor 분석"""
        print("\n" + "=" * 70)
        print("[3] Market Factor 데이터 포인트 분석")
        print("=" * 70)

        results = []
        days_per_year = self.config.trading_days_per_year

        for years in range(1, self.config.max_years + 1):
            days = years * days_per_year
            code_counts = self.df.groupby('코드').head(days).groupby('코드').size()
            codes_with_enough = code_counts[code_counts >= days]
            num_codes = len(codes_with_enough)

            results.append({
                '기간(년)': years,
                'Unique 코드 개수': num_codes,
                'Unique 코드 개수 × 기간': num_codes * years
            })

        result_df = pd.DataFrame(results)

        # 그래프 저장
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(result_df['기간(년)'], result_df['Unique 코드 개수 × 기간'],
                marker='o', linewidth=2, markersize=8, color='#2E86AB')
        ax.set_xlabel('기간 (년)', fontsize=12)
        ax.set_ylabel('Unique 코드 개수 × 기간', fontsize=12)
        ax.set_title('기간별 충분한 데이터를 가진 상품 수 × 기간', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, self.config.max_years + 1))
        plt.tight_layout()

        fig_path = os.path.join(self.config.output_dir, 'DataPoints_for_MarketFactor.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        # CSV 저장
        csv_path = os.path.join(self.config.output_dir, 'DataPoints_for_MarketFactor.csv')
        result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

        print(f"\n  결과:")
        print(result_df.to_string(index=False))
        print(f"\n  -> 저장: DataPoints_for_MarketFactor.csv/png")

        return result_df

    def analyze_by_category(self, category: str) -> Dict[str, pd.DataFrame]:
        """카테고리별 분석"""
        print(f"\n  [{category}] 분석 중...")

        # 상품 정보 병합
        df_merged = self.df.merge(
            self.df_product[['상품 코드', category]],
            left_on='코드', right_on='상품 코드', how='left'
        )

        results_by_sub = {}
        days_per_year = self.config.trading_days_per_year

        for sub in df_merged[category].dropna().unique():
            df_sub = df_merged[df_merged[category] == sub].copy()
            results = []

            for years in range(1, self.config.max_years + 1):
                days = years * days_per_year
                code_counts = df_sub.groupby('코드').head(days).groupby('코드').size()
                codes_with_enough = code_counts[code_counts >= days]
                num_codes = len(codes_with_enough)

                results.append({
                    '기간(년)': years,
                    'Unique 코드 개수': num_codes,
                    'Unique 코드 개수 × 기간': num_codes * years
                })

            results_by_sub[sub] = pd.DataFrame(results)

        return results_by_sub

    def analyze_all_categories(self) -> None:
        """모든 카테고리 분석"""
        print("\n" + "=" * 70)
        print("[4] 카테고리별 데이터 포인트 분석")
        print("=" * 70)

        for category in self.config.categories:
            results_dict = self.analyze_by_category(category)

            # 그래프 그리기
            fig, ax = plt.subplots(figsize=(12, 7))

            for sub, result_df in results_dict.items():
                ax.plot(result_df['기간(년)'], result_df['Unique 코드 개수 × 기간'],
                        marker='o', linewidth=2, markersize=6, label=str(sub))

            ax.set_xlabel('기간 (년)', fontsize=12)
            ax.set_ylabel('Unique 코드 개수 × 기간', fontsize=12)
            ax.set_title(f'{category}별 기간별 데이터 포인트', fontsize=13, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xticks(range(1, self.config.max_years + 1))
            ax.legend(loc='best', fontsize=10)
            plt.tight_layout()

            fig_path = os.path.join(self.config.output_dir, f'DataPoints_by_{category}.png')
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            plt.close()

            # CSV 저장
            combined_df = pd.DataFrame()
            for sub, result_df in results_dict.items():
                temp = result_df.copy()
                temp[category] = sub
                combined_df = pd.concat([combined_df, temp], ignore_index=True)

            cols = [category, '기간(년)', 'Unique 코드 개수', 'Unique 코드 개수 × 기간']
            combined_df = combined_df[cols].sort_values([category, '기간(년)'])

            csv_path = os.path.join(self.config.output_dir, f'DataPoints_by_{category}.csv')
            combined_df.to_csv(csv_path, index=False, encoding='utf-8-sig')

            print(f"    -> 저장: DataPoints_by_{category}.csv/png")

    def run(self) -> None:
        """전체 분석 실행"""
        print("\n" + "#" * 70)
        print("#  분석기간 설정 및 데이터 포인트 분석")
        print("#" * 70)

        self.load_data()
        self.verify_code_matching()
        self.analyze_market_factor()
        self.analyze_all_categories()

        print("\n" + "=" * 70)
        print("  분석 완료!")
        print(f"  결과 저장 위치: {self.config.output_dir}")
        print("=" * 70)


def run_analysis_period_check():
    """분석기간 체크 실행 함수 (main.py에서 호출용)"""
    checker = AnalysisPeriodChecker()
    checker.run()
    return checker


if __name__ == "__main__":
    run_analysis_period_check()
