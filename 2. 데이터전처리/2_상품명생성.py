# -*- coding: utf-8 -*-
"""
상품명.xlsx 생성 스크립트
=========================
수익률 계산 결과를 기반으로 4_상품분류_LLM.py의 input 파일을 생성합니다.

생성 컬럼:
    - 상품 코드: 수익률 결과의 '코드'
    - 상품명: 수익률 결과의 '종목명'
    - 비고: 수익률 결과의 '자산군' (ETF/FUND/REITs)
    - 최초기준일자: 각 상품별 데이터 존재 첫 날짜
    - 투자한도: ETF→etf_code_list, FUND→fund_code_list, REITs→0.7
    - TDF: 상품명에 'TDF' 포함 여부

사용법:
    1. config.py에서 ProductNameConfig 설정 확인
    2. python 2_상품명생성.py 실행
"""

import pandas as pd

from config import ProductNameConfig


class ProductNameGenerator:
    """상품명.xlsx 생성 클래스"""

    def __init__(self, config: ProductNameConfig = None):
        self.config = config or ProductNameConfig()
        self.return_df = None
        self.product_df = None

    def load_return_data(self) -> pd.DataFrame:
        """수익률 결과 데이터 로드"""
        print("[1] 수익률 결과 데이터 로드")

        self.return_df = pd.read_csv(
            self.config.path_excess_return, encoding="utf-8-sig"
        )
        self.return_df["기준일자"] = pd.to_datetime(self.return_df["기준일자"])
        self.return_df["코드"] = self.return_df["코드"].astype(str).str.strip()

        print(f"    총 {len(self.return_df):,}행, {self.return_df['코드'].nunique():,}개 상품")
        return self.return_df

    def extract_product_info(self) -> pd.DataFrame:
        """상품별 기본 정보 추출"""
        print("[2] 상품별 정보 추출")

        self.product_df = self.return_df.groupby("코드").agg({
            "종목명": "last",
            "자산군": "first",
            "기준일자": "min"
        }).reset_index()

        self.product_df.columns = ["상품 코드", "상품명", "비고", "최초기준일자"]

        print(f"    {len(self.product_df):,}개 상품 추출")
        return self.product_df

    def load_etf_investment_limit(self) -> pd.DataFrame:
        """ETF 투자한도 로드"""
        print("[3] ETF 투자한도 로드")

        df = pd.read_excel(self.config.path_etf_code, skiprows=1)
        df = df.rename(columns={df.columns[1]: "코드", df.columns[-1]: "투자한도"})
        df["코드"] = df["코드"].astype(str).str.strip()
        df["투자한도"] = pd.to_numeric(df["투자한도"], errors="coerce")

        result = df[["코드", "투자한도"]].dropna(subset=["코드"])
        print(f"    ETF {len(result):,}개 로드")
        return result

    def load_fund_investment_limit(self) -> pd.DataFrame:
        """FUND 투자한도 로드"""
        print("[4] FUND 투자한도 로드")

        df = pd.read_excel(self.config.path_fund_code, skiprows=[1])
        df = df.rename(columns={"코드": "코드", "퇴직연금 투자한도": "투자한도"})
        df["코드"] = df["코드"].astype(str).str.strip()
        df["투자한도"] = pd.to_numeric(df["투자한도"], errors="coerce")

        result = df[["코드", "투자한도"]].dropna(subset=["코드"])
        print(f"    FUND {len(result):,}개 로드")
        return result

    def merge_investment_limit(self, etf_limit: pd.DataFrame, fund_limit: pd.DataFrame) -> None:
        """투자한도 매칭"""
        print("[5] 투자한도 매칭")

        self.product_df["투자한도"] = None

        # ETF
        etf_mask = self.product_df["비고"] == "ETF"
        etf_limit_dict = etf_limit.set_index("코드")["투자한도"].to_dict()
        self.product_df.loc[etf_mask, "투자한도"] = (
            self.product_df.loc[etf_mask, "상품 코드"].map(etf_limit_dict)
        )
        etf_matched = self.product_df.loc[etf_mask, "투자한도"].notna().sum()
        print(f"    ETF 매칭: {etf_matched}/{etf_mask.sum()}")

        # FUND
        fund_mask = self.product_df["비고"] == "FUND"
        fund_limit_dict = fund_limit.set_index("코드")["투자한도"].to_dict()
        self.product_df.loc[fund_mask, "투자한도"] = (
            self.product_df.loc[fund_mask, "상품 코드"].map(fund_limit_dict)
        )
        fund_matched = self.product_df.loc[fund_mask, "투자한도"].notna().sum()
        print(f"    FUND 매칭: {fund_matched}/{fund_mask.sum()}")

        # REITs
        reits_mask = self.product_df["비고"] == "REITs"
        self.product_df.loc[reits_mask, "투자한도"] = self.config.reits_default_limit
        print(f"    REITs: {reits_mask.sum()}개 ({self.config.reits_default_limit} 고정)")

    def add_tdf_flag(self) -> None:
        """TDF 플래그 추가 (연도 추출)"""
        print("[6] TDF 플래그 추가")
        import re

        def extract_tdf_year(name: str) -> int:
            """상품명에서 TDF 연도 추출"""
            if pd.isna(name):
                return 0
            name_upper = str(name).upper()
            if "TDF" not in name_upper:
                return 0
            # 20XX 형식 연도 추출 (2020~2060)
            match = re.search(r"20[2-6][0-9]", name_upper)
            if match:
                return int(match.group())
            return 0

        self.product_df["TDF"] = self.product_df["상품명"].apply(extract_tdf_year)

        tdf_count = (self.product_df["TDF"] > 0).sum()
        print(f"    TDF 상품: {tdf_count}개")

    def save_result(self) -> None:
        """결과 저장"""
        print("[7] 결과 저장")

        output_cols = ["상품 코드", "상품명", "비고", "최초기준일자", "투자한도", "TDF"]
        result = self.product_df[output_cols]

        result.to_excel(self.config.output_file, index=False, engine="openpyxl")
        print(f"    저장 완료: {self.config.output_file}")
        print(f"    총 {len(result):,}개 상품")

    def run(self) -> pd.DataFrame:
        """전체 실행"""
        print("=" * 60)
        print("상품명.xlsx 생성")
        print("=" * 60)
        print()

        self.load_return_data()
        self.extract_product_info()

        etf_limit = self.load_etf_investment_limit()
        fund_limit = self.load_fund_investment_limit()

        self.merge_investment_limit(etf_limit, fund_limit)
        self.add_tdf_flag()
        self.save_result()

        print()
        print("=" * 60)
        print("완료")
        print("=" * 60)

        # 요약
        print()
        print("[자산군별 상품 수]")
        print(self.product_df["비고"].value_counts().to_string())
        print()
        print("[투자한도 매칭 현황]")
        print(f"  매칭 성공: {self.product_df['투자한도'].notna().sum()}")
        print(f"  매칭 실패: {self.product_df['투자한도'].isna().sum()}")

        return self.product_df


def run_product_name_generation():
    """상품명 생성 실행 함수 (main.py에서 호출용)"""
    generator = ProductNameGenerator()
    return generator.run()


if __name__ == "__main__":
    run_product_name_generation()
