# -*- coding: utf-8 -*-
"""
수익률 계산 파이프라인
======================
ETF, 펀드, REITs 데이터를 통합하여 분배금 반영 수익률 및 초과수익률을 계산합니다.

처리 내용:
    1. 가격 데이터 로드 및 표준화
    2. 이상치 날짜 필터링 (펀드 수 급감일 제거)
    3. 분배금 데이터 로드 및 배당락일 변환
    4. 총수익률 계산 (분배금 반영)
    5. 초과수익률 계산 (무위험수익률 차감)
    6. 상품별 요약 통계 생성

사용법:
    1. config.py에서 ReturnConfig 설정 확인
    2. python 2_수익률계산.py 실행
"""

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from config import ReturnConfig


# =============================================================================
# 유틸리티 클래스
# =============================================================================

class DateParser:
    """날짜 파싱 유틸리티"""

    @staticmethod
    def parse_mixed(s: pd.Series) -> pd.Series:
        """혼합 날짜 형식 파싱"""
        if pd.api.types.is_datetime64_any_dtype(s):
            dt = pd.to_datetime(s, errors="coerce")
            try:
                dt = dt.dt.tz_localize(None)
            except Exception:
                try:
                    dt = dt.dt.tz_convert(None)
                except Exception:
                    pass
            return dt.dt.normalize()

        if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
            v = pd.to_numeric(s, errors="coerce").astype("Int64")
            dt = pd.to_datetime(v.astype("string"), format="%Y%m%d", errors="coerce")
            return dt.dt.normalize()

        s_str = s.astype("string").str.strip()
        mask_8 = s_str.str.fullmatch(r"\d{8}")
        dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

        if mask_8.any():
            dt.loc[mask_8] = pd.to_datetime(s_str[mask_8], format="%Y%m%d", errors="coerce")
        rem = ~mask_8
        if rem.any():
            dt.loc[rem] = pd.to_datetime(s_str[rem], errors="coerce")

        try:
            dt = dt.dt.tz_localize(None)
        except Exception:
            try:
                dt = dt.dt.tz_convert(None)
            except Exception:
                pass

        return dt.dt.normalize()

    @staticmethod
    def parse_dot_or_dash(s: pd.Series) -> pd.Series:
        """점/대시 형식 날짜 파싱"""
        s = s.astype("string").str.strip()
        mask_dot = s.str.contains(r"\.", na=False)
        dt = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")

        if mask_dot.any():
            dt.loc[mask_dot] = pd.to_datetime(s[mask_dot], format="%Y.%m.%d", errors="coerce")
        mask_dash = ~mask_dot
        if mask_dash.any():
            dt.loc[mask_dash] = pd.to_datetime(s[mask_dash], errors="coerce")

        return dt.dt.normalize()


class CodeNormalizer:
    """코드 정규화 유틸리티"""

    @staticmethod
    def normalize(code: str) -> str:
        """코드 정규화 (4~5자리 숫자는 6자리로)"""
        code_str = str(code).strip()
        if code_str.isdigit() and len(code_str) in [4, 5]:
            return code_str.zfill(6)
        return code_str


# =============================================================================
# 데이터 로더 클래스
# =============================================================================

class BaseDataLoader(ABC):
    """데이터 로더 추상 클래스"""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        pass


class PriceDataLoader(BaseDataLoader):
    """가격 데이터 로더"""

    def __init__(
        self,
        path: str,
        usecols: List[str],
        col_mapping: Dict[str, str],
        asset_label: str
    ):
        self.path = path
        self.usecols = usecols
        self.col_mapping = col_mapping
        self.asset_label = asset_label

    def load(self) -> pd.DataFrame:
        """데이터 로드 및 표준화"""
        dtype_map = {
            c: "string" for c in self.usecols
            if any(k in c for k in ["기준일자", "거래일", "코드", "명"])
        }

        df = pd.read_csv(
            self.path,
            usecols=self.usecols,
            dtype=dtype_map,
            encoding="utf-8-sig",
            low_memory=False
        )

        # 컬럼명 표준화
        df = df.rename(columns=self.col_mapping)

        # 날짜 파싱
        df["기준일자"] = DateParser.parse_mixed(df["기준일자"])

        # 코드 정규화
        df["코드"] = df["코드"].apply(CodeNormalizer.normalize)

        # 종목명 문자열 변환
        df["종목명"] = df["종목명"].astype("string")

        # 가격 숫자 변환
        df["가격"] = pd.to_numeric(df["가격"], errors="coerce")

        # 자산군 라벨
        df["자산군"] = self.asset_label

        # 중복 제거
        df = df.drop_duplicates(subset=["코드", "기준일자"], keep="last")

        # 유효하지 않은 가격 처리
        df.loc[df["가격"] <= 0, "가격"] = np.nan

        # 정렬
        df = df.sort_values(["코드", "기준일자"]).reset_index(drop=True)

        return df


class DividendDataLoader(BaseDataLoader):
    """배당 데이터 로더"""

    def __init__(
        self,
        path: str,
        col_mapping: Dict[str, str],
        asset_label: str,
        multiplier: float = 1.0
    ):
        self.path = path
        self.col_mapping = col_mapping
        self.asset_label = asset_label
        self.multiplier = multiplier

    def load(self) -> pd.DataFrame:
        """배당 데이터 로드"""
        # 코드 컬럼 찾기
        code_col = [k for k, v in self.col_mapping.items() if v == "코드"][0]

        df = pd.read_excel(self.path, dtype={code_col: str})
        df = df.rename(columns=self.col_mapping)

        df["코드"] = df["코드"].apply(CodeNormalizer.normalize)
        df["권리기준일"] = pd.to_datetime(df["권리기준일"])
        df["배당금"] = pd.to_numeric(df["배당금"], errors="coerce").fillna(0) * self.multiplier
        df["자산군"] = self.asset_label

        # 필요한 컬럼만 선택 및 집계
        df = df[["코드", "권리기준일", "배당금", "자산군"]].copy()
        df = df.groupby(["코드", "권리기준일", "자산군"], as_index=False)["배당금"].sum()

        return df


class RiskFreeRateLoader(BaseDataLoader):
    """무위험수익률 로더"""

    def __init__(self, path: str, trading_days: int):
        self.path = path
        self.trading_days = trading_days

    def load(self) -> pd.DataFrame:
        """무위험수익률 로드 및 일별 변환"""
        df = pd.read_excel(self.path)
        df = df.rename(columns={"거래일": "기준일자", "무위험수익률": "무위험수익률(연율%)"})

        df["기준일자"] = DateParser.parse_dot_or_dash(df["기준일자"])
        df["무위험수익률(연율%)"] = pd.to_numeric(df["무위험수익률(연율%)"], errors="coerce")

        df = df.dropna(subset=["기준일자"])
        df = df.drop_duplicates(subset=["기준일자"], keep="last")
        df = df.sort_values("기준일자")

        # 일별 수익률 변환 (단리 방식)
        rf_annual = df["무위험수익률(연율%)"] / 100.0
        df["무위험수익률"] = rf_annual / self.trading_days

        return df[["기준일자", "무위험수익률"]].copy()


# =============================================================================
# 데이터 처리 클래스
# =============================================================================

class AnomalyDateDetector:
    """이상치 날짜 탐지"""

    def __init__(self, threshold: float = -0.1, window: int = 30):
        self.threshold = threshold
        self.window = window

    def detect(self, df: pd.DataFrame) -> List[pd.Timestamp]:
        """이상치 날짜 탐지"""
        daily_count = df.groupby("기준일자")["코드"].nunique().reset_index()
        daily_count.columns = ["날짜", "상품수"]
        daily_count = daily_count.sort_values("날짜").reset_index(drop=True)

        daily_count["이동평균"] = daily_count["상품수"].rolling(
            window=self.window, min_periods=1
        ).mean()
        daily_count["편차율"] = (
            (daily_count["상품수"] - daily_count["이동평균"]) / daily_count["이동평균"]
        )

        anomalies = daily_count[daily_count["편차율"] < self.threshold]
        return anomalies["날짜"].tolist()


class ExDividendDateCalculator:
    """배당락일 계산"""

    def __init__(self, price_data: Dict[str, pd.DataFrame]):
        self.trading_dates = {}
        self.return_dicts = {}

        for asset, df in price_data.items():
            self.trading_dates[asset] = np.sort(df["기준일자"].unique())
            self.return_dicts[asset] = self._build_return_dict(df)

    def _build_return_dict(self, df: pd.DataFrame) -> Dict[Tuple[str, pd.Timestamp], float]:
        """수익률 딕셔너리 생성"""
        df = df.sort_values(["코드", "기준일자"]).copy()
        df["전일가격"] = df.groupby("코드")["가격"].shift(1)
        df["일별수익률"] = (df["가격"] - df["전일가격"]) / df["전일가격"]

        return_dict = {}
        for _, row in df.iterrows():
            if pd.notna(row["일별수익률"]):
                return_dict[(row["코드"], row["기준일자"])] = row["일별수익률"]

        return return_dict

    def calculate_for_etf_reits(
        self,
        row: pd.Series,
        asset: str,
        default_offset: int
    ) -> pd.Timestamp:
        """ETF/REITs용 배당락일 계산 (T-1, T-2 중 하락 큰 날)"""
        code = row["코드"]
        record_date = row["권리기준일"]

        trading_values = self.trading_dates[asset].astype("datetime64[ns]")
        record_value = np.datetime64(record_date, "ns")

        if record_value > trading_values[-1]:
            return pd.NaT

        idx = np.searchsorted(trading_values, record_value, side="left")

        t1_idx = idx - 1
        t2_idx = idx - 2

        t1_date = None if t1_idx < 0 else trading_values[t1_idx]
        t2_date = None if t2_idx < 0 else trading_values[t2_idx]

        return_dict = self.return_dicts[asset]
        t1_return = return_dict.get((code, pd.Timestamp(t1_date))) if t1_date is not None else None
        t2_return = return_dict.get((code, pd.Timestamp(t2_date))) if t2_date is not None else None

        if t1_return is not None and t2_return is not None:
            if t1_return < t2_return:
                return pd.Timestamp(t1_date)
            elif t2_return < t1_return:
                return pd.Timestamp(t2_date)
            else:
                if default_offset == 1 and t1_date is not None:
                    return pd.Timestamp(t1_date)
                elif default_offset == 2 and t2_date is not None:
                    return pd.Timestamp(t2_date)
                elif t1_date is not None:
                    return pd.Timestamp(t1_date)
                return pd.NaT
        elif t1_return is not None:
            return pd.Timestamp(t1_date)
        elif t2_return is not None:
            return pd.Timestamp(t2_date)
        else:
            if default_offset == 1 and t1_date is not None:
                return pd.Timestamp(t1_date)
            elif default_offset == 2 and t2_date is not None:
                return pd.Timestamp(t2_date)
            return pd.NaT

    def calculate_for_fund(self, row: pd.Series) -> Tuple[pd.Timestamp, Optional[int]]:
        """펀드용 배당락일 계산 (T-3 ~ T+3 범위)"""
        code = row["코드"]
        record_date = row["권리기준일"]

        trading_values = self.trading_dates["FUND"].astype("datetime64[ns]")
        record_value = np.datetime64(record_date, "ns")

        if record_value > trading_values[-1]:
            return pd.NaT, None

        idx = np.searchsorted(trading_values, record_value, side="left")

        return_dict = self.return_dicts["FUND"]
        candidates = []

        for offset in range(-3, 4):
            t_idx = idx + offset
            if 0 <= t_idx < len(trading_values):
                t_date = trading_values[t_idx]
                t_return = return_dict.get((code, pd.Timestamp(t_date)))
                if t_return is not None:
                    candidates.append({
                        "offset": offset,
                        "date": t_date,
                        "return": t_return
                    })

        if not candidates:
            t_plus1_idx = idx + 1
            if 0 <= t_plus1_idx < len(trading_values):
                return pd.Timestamp(trading_values[t_plus1_idx]), 1
            if 0 <= idx < len(trading_values):
                return pd.Timestamp(trading_values[idx]), 0
            return pd.NaT, None

        best = min(candidates, key=lambda x: x["return"])
        return pd.Timestamp(best["date"]), best["offset"]


class ReturnCalculator:
    """수익률 계산"""

    def __init__(self, trading_days: int):
        self.trading_days = trading_days

    def compute_total_return(self, df: pd.DataFrame) -> pd.DataFrame:
        """총수익률 계산 (분배금 반영)"""
        df = df.sort_values(["코드", "기준일자"]).reset_index(drop=True)

        df["전일가격"] = df.groupby("코드")["가격"].shift(1)
        df["총액"] = df["가격"] + df["배당금"]

        valid_mask = (df["전일가격"] > 0) & (df["총액"] > 0)

        df["수익률"] = np.nan
        df.loc[valid_mask, "수익률"] = (
            (df.loc[valid_mask, "총액"] - df.loc[valid_mask, "전일가격"])
            / df.loc[valid_mask, "전일가격"]
        )

        df = df.drop(columns=["전일가격", "총액"])
        return df

    def compute_excess_return(
        self,
        df: pd.DataFrame,
        rf_df: pd.DataFrame
    ) -> pd.DataFrame:
        """초과수익률 계산"""
        # 전체 날짜에 대한 무위험수익률 매핑
        full_dates = pd.DataFrame({"기준일자": sorted(df["기준일자"].unique())})
        rf_full = pd.merge(full_dates, rf_df, on="기준일자", how="left").sort_values("기준일자")
        rf_full["무위험수익률"] = rf_full["무위험수익률"].ffill()

        # 병합 및 초과수익률 계산
        merged = pd.merge(df, rf_full, on="기준일자", how="left")
        merged["초과수익률"] = merged["수익률"] - merged["무위험수익률"]

        return merged


class SummaryStatisticsCalculator:
    """요약 통계 계산"""

    def __init__(self, trading_days: int):
        self.trading_days = trading_days

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """상품별 요약 통계 계산"""
        # 기본 정보
        product_info = df.groupby("코드").agg({
            "종목명": "first",
            "자산군": "first",
            "기준일자": ["min", "max", "count"]
        }).reset_index()
        product_info.columns = ["코드", "종목명", "자산군", "시작일", "종료일", "데이터수"]

        # 수익률 통계
        return_stats = df.groupby("코드")["수익률"].agg([
            ("평균", "mean"),
            ("표준편차", "std"),
            ("최소", "min"),
            ("Q1_25%", lambda x: x.quantile(0.25)),
            ("중앙값", "median"),
            ("Q3_75%", lambda x: x.quantile(0.75)),
            ("최대", "max"),
            ("왜도", "skew"),
            ("첨도", lambda x: x.kurtosis()),
            ("양수비율", lambda x: (x > 0).mean())
        ]).reset_index()

        # 초과수익률 통계
        excess_stats = df.groupby("코드")["초과수익률"].agg([
            ("초과_평균", "mean"),
            ("초과_표준편차", "std"),
            ("초과_최소", "min"),
            ("초과_Q1_25%", lambda x: x.quantile(0.25)),
            ("초과_중앙값", "median"),
            ("초과_Q3_75%", lambda x: x.quantile(0.75)),
            ("초과_최대", "max"),
            ("초과_왜도", "skew"),
            ("초과_첨도", lambda x: x.kurtosis()),
            ("초과_양수비율", lambda x: (x > 0).mean())
        ]).reset_index()

        # 누적수익률
        cumret = df.groupby("코드").agg({
            "수익률": lambda x: (1 + x).prod() - 1,
            "초과수익률": lambda x: (1 + x).prod() - 1
        }).reset_index()
        cumret.columns = ["코드", "누적수익률", "누적초과수익률"]

        # 연율화 수익률
        def annualized_return(group):
            n_years = len(group) / self.trading_days
            if n_years > 0:
                total_return = (1 + group).prod()
                return total_return ** (1 / n_years) - 1
            return np.nan

        annual = df.groupby("코드").agg({
            "수익률": annualized_return,
            "초과수익률": annualized_return
        }).reset_index()
        annual.columns = ["코드", "연율화수익률", "연율화초과수익률"]

        # 연율화 변동성
        vol = df.groupby("코드").agg({
            "수익률": lambda x: x.std() * np.sqrt(self.trading_days),
            "초과수익률": lambda x: x.std() * np.sqrt(self.trading_days)
        }).reset_index()
        vol.columns = ["코드", "연율화변동성", "연율화초과변동성"]

        # 샤프비율
        def calc_sharpe(group):
            if group["초과수익률"].std() > 0:
                annual_excess = annualized_return(group["초과수익률"])
                annual_vol = group["초과수익률"].std() * np.sqrt(self.trading_days)
                return annual_excess / annual_vol
            return np.nan

        sharpe = df.groupby("코드").apply(calc_sharpe, include_groups=False).reset_index()
        sharpe.columns = ["코드", "샤프비율"]

        # MDD
        def calc_mdd(group):
            group = group.sort_values("기준일자")
            cumulative = (1 + group["수익률"]).cumprod()
            running_max = cumulative.cummax()
            drawdown = (cumulative - running_max) / running_max
            return drawdown.min()

        mdd = df.groupby("코드").apply(calc_mdd, include_groups=False).reset_index()
        mdd.columns = ["코드", "MDD"]

        # 분배금 통계
        div_stats = df.groupby("코드").agg({
            "분배금": ["sum", lambda x: (x > 0).sum()]
        }).reset_index()
        div_stats.columns = ["코드", "분배금합계", "분배금횟수"]

        # 전체 병합
        summary = product_info.copy()
        for stats_df in [return_stats, excess_stats, cumret, annual, vol, sharpe, mdd, div_stats]:
            summary = summary.merge(stats_df, on="코드", how="left")

        # 정렬 및 반올림
        summary = summary.sort_values(["자산군", "코드"]).reset_index(drop=True)
        numeric_cols = summary.select_dtypes(include=[np.number]).columns
        summary[numeric_cols] = summary[numeric_cols].round(6)

        return summary


# =============================================================================
# 메인 파이프라인
# =============================================================================

class ReturnCalculationPipeline:
    """수익률 계산 파이프라인"""

    def __init__(self, config: ReturnConfig):
        self.config = config
        self.anomaly_detector = AnomalyDateDetector(config.anomaly_threshold)
        self.return_calculator = ReturnCalculator(config.trading_days_per_year)
        self.summary_calculator = SummaryStatisticsCalculator(config.trading_days_per_year)

    def load_price_data(self) -> Dict[str, pd.DataFrame]:
        """가격 데이터 로드"""
        print("=" * 70)
        print("[1] 가격 데이터 로드")
        print("=" * 70)

        loaders = {
            "ETF": PriceDataLoader(
                self.config.path_etf,
                ["기준일자", "종목코드", "종목명", "순자산가치"],
                {"종목코드": "코드", "순자산가치": "가격"},
                "ETF"
            ),
            "REITs": PriceDataLoader(
                self.config.path_reits,
                ["기준일자", "종목코드", "종목명", "종가"],
                {"종목코드": "코드", "종가": "가격"},
                "REITs"
            ),
            "FUND": PriceDataLoader(
                self.config.path_fund,
                ["기준일자", "펀드코드", "펀드명", "기준가격"],
                {"펀드코드": "코드", "펀드명": "종목명", "기준가격": "가격"},
                "FUND"
            )
        }

        data = {}
        for name, loader in loaders.items():
            df = loader.load()
            print(f"  {name}: {len(df):,}행 로드")
            data[name] = df

        return data

    def filter_anomaly_dates(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """이상치 날짜 필터링"""
        print("\n" + "=" * 70)
        print("[2] 이상치 날짜 필터링")
        print("=" * 70)

        # 임시 통합
        combined = pd.concat([
            df[["기준일자", "코드"]] for df in data.values()
        ], ignore_index=True)

        # 이상치 탐지
        anomaly_dates = self.anomaly_detector.detect(combined)
        print(f"  이상치 날짜 수: {len(anomaly_dates)}개")

        # 필터링
        filtered_data = {}
        for name, df in data.items():
            before = len(df)
            filtered = df[~df["기준일자"].isin(anomaly_dates)].copy()
            print(f"  {name}: {before:,} -> {len(filtered):,}행")
            filtered_data[name] = filtered

        return filtered_data

    def filter_excluded_codes(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """제외 상품 필터링"""
        print("\n" + "=" * 70)
        print("[3] 제외 상품 필터링")
        print("=" * 70)

        filtered_data = {}
        for name, df in data.items():
            before = len(df)
            filtered = df[~df["코드"].isin(self.config.excluded_codes)].copy()
            removed = before - len(filtered)
            if removed > 0:
                print(f"  {name}: {removed:,}행 제거")
            filtered_data[name] = filtered

        return filtered_data

    def load_dividend_data(self) -> pd.DataFrame:
        """배당 데이터 로드"""
        print("\n" + "=" * 70)
        print("[4] 배당 데이터 로드")
        print("=" * 70)

        dfs = []

        # ETF 배당: 종목명 기준으로 가격 파일에서 종목코드 매핑
        etf_div_df = pd.read_excel(self.config.path_etf_div)
        etf_price_df = pd.read_csv(self.config.path_etf, encoding="utf-8-sig", low_memory=False)

        # 종목명 → 종목코드 매핑 (가장 최근 데이터 기준)
        name_to_code = (
            etf_price_df.sort_values("기준일자")
            .drop_duplicates(subset=["종목명"], keep="last")
            .set_index("종목명")["종목코드"]
            .to_dict()
        )

        etf_div_df["코드"] = etf_div_df["종목명"].map(name_to_code)
        etf_div_df["권리기준일"] = pd.to_datetime(etf_div_df["권리기준일"])
        etf_div_df["배당금"] = pd.to_numeric(etf_div_df["주당분배금"], errors="coerce").fillna(0)
        etf_div_df["자산군"] = "ETF"
        etf_div_df["코드"] = etf_div_df["코드"].apply(CodeNormalizer.normalize)

        # 매핑 안된 건 제거
        before = len(etf_div_df)
        etf_div_df = etf_div_df.dropna(subset=["코드"])
        etf_div_df = etf_div_df[["코드", "권리기준일", "배당금", "자산군"]]
        etf_div_df = etf_div_df.groupby(["코드", "권리기준일", "자산군"], as_index=False)["배당금"].sum()
        print(f"  ETF: {len(etf_div_df):,}건 (매핑실패 제거: {before - len(etf_div_df):,}건)")
        dfs.append(etf_div_df)

        # FUND, REITs 배당
        other_loaders = [
            DividendDataLoader(
                self.config.path_fund_div,
                {"펀드코드": "코드", "기준일자": "권리기준일", "주좌당배당액": "배당금"},
                "FUND",
                multiplier=1000.0
            ),
            DividendDataLoader(
                self.config.path_reits_div,
                {"종목코드": "코드", "배정기준일": "권리기준일", "주당배당액": "배당금"},
                "REITs"
            )
        ]

        for loader in other_loaders:
            df = loader.load()
            print(f"  {loader.asset_label}: {len(df):,}건")
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def calculate_ex_dividend_dates(
        self,
        div_df: pd.DataFrame,
        price_data: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """배당락일 계산"""
        print("\n" + "=" * 70)
        print("[5] 배당락일 계산")
        print("=" * 70)

        calculator = ExDividendDateCalculator(price_data)

        div_df["배당락일"] = pd.NaT

        # ETF, REITs
        for asset, default_off in [("ETF", 2), ("REITs", 1)]:
            mask = div_df["자산군"] == asset
            if mask.any():
                div_df.loc[mask, "배당락일"] = div_df.loc[mask].apply(
                    lambda row: calculator.calculate_for_etf_reits(row, asset, default_off),
                    axis=1
                )
                valid = div_df.loc[mask, "배당락일"].notna().sum()
                print(f"  {asset}: {valid:,}건 변환 완료")

        # FUND
        mask = div_df["자산군"] == "FUND"
        if mask.any():
            results = div_df.loc[mask].apply(
                lambda row: calculator.calculate_for_fund(row),
                axis=1
            )
            div_df.loc[mask, "배당락일"] = results.apply(lambda x: x[0])
            valid = div_df.loc[mask, "배당락일"].notna().sum()
            print(f"  FUND: {valid:,}건 변환 완료")

        # 유효한 배당락일만 유지
        before = len(div_df)
        div_df = div_df.dropna(subset=["배당락일"])
        print(f"  최종: {len(div_df):,}건 (제거: {before - len(div_df):,}건)")

        return div_df

    def merge_dividend(
        self,
        price_data: Dict[str, pd.DataFrame],
        div_df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """가격과 배당 데이터 병합"""
        print("\n" + "=" * 70)
        print("[6] 가격 + 배당 병합")
        print("=" * 70)

        div_for_merge = div_df[["코드", "배당락일", "배당금", "자산군"]].copy()
        div_for_merge = div_for_merge.rename(columns={"배당락일": "기준일자"})
        div_for_merge = div_for_merge.groupby(
            ["코드", "기준일자", "자산군"], as_index=False
        )["배당금"].sum()

        merged_data = {}
        for name, df in price_data.items():
            div_subset = div_for_merge[div_for_merge["자산군"] == name][
                ["코드", "기준일자", "배당금"]
            ]
            merged = df.merge(div_subset, on=["코드", "기준일자"], how="left")
            merged["배당금"] = merged["배당금"].fillna(0)
            matched = (merged["배당금"] > 0).sum()
            print(f"  {name}: 배당 매칭 {matched:,}건")
            merged_data[name] = merged

        return merged_data

    def calculate_returns(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """수익률 계산"""
        print("\n" + "=" * 70)
        print("[7] 수익률 계산")
        print("=" * 70)

        for name, df in data.items():
            data[name] = self.return_calculator.compute_total_return(df)
            valid = data[name]["수익률"].notna().sum()
            print(f"  {name}: {valid:,}건")

        # 통합
        cols_out = ["기준일자", "코드", "종목명", "가격", "배당금", "수익률", "자산군"]
        combined = pd.concat([df[cols_out] for df in data.values()], ignore_index=True)
        combined = combined.dropna(subset=["수익률"])
        combined = combined.sort_values(["자산군", "코드", "기준일자"]).reset_index(drop=True)

        print(f"  총: {len(combined):,}행")
        return combined

    def unify_product_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """종목명 통일"""
        print("\n" + "=" * 70)
        print("[8] 종목명 통일")
        print("=" * 70)

        latest_names = (
            df.sort_values("기준일자")
            .groupby("코드")["종목명"]
            .last()
            .to_dict()
        )
        df["종목명"] = df["코드"].map(latest_names)
        print("  완료")

        return df

    def calculate_excess_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """초과수익률 계산"""
        print("\n" + "=" * 70)
        print("[9] 초과수익률 계산")
        print("=" * 70)

        rf_loader = RiskFreeRateLoader(
            self.config.path_rf,
            self.config.trading_days_per_year
        )
        rf_df = rf_loader.load()
        print(f"  무위험수익률 데이터: {len(rf_df):,}일")

        df = self.return_calculator.compute_excess_return(df, rf_df)
        print("  계산 완료")

        return df

    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """데이터 필터링"""
        print("\n" + "=" * 70)
        print("[10] 데이터 필터링")
        print("=" * 70)

        # 기준일 필터링 (cutoff_date가 None이면 건너뜀)
        if self.config.cutoff_date:
            cutoff = pd.Timestamp(self.config.cutoff_date)
            before = len(df)
            df = df[df["기준일자"] > cutoff].copy()
            print(f"  {self.config.cutoff_date} 이전 제거: {before:,} -> {len(df):,}행")
        else:
            print("  기준일 필터링: 없음 (전체 데이터 사용)")

        # 최소 데이터 수 필터링
        product_counts = df.groupby("코드").size()
        valid_codes = product_counts[product_counts >= self.config.min_data_points].index

        before_products = df["코드"].nunique()
        before_rows = len(df)
        df = df[df["코드"].isin(valid_codes)].copy()

        print(f"  {self.config.min_data_points}일 미만 제거: {before_products:,} -> {df['코드'].nunique():,}개 상품")
        print(f"    데이터: {before_rows:,} -> {len(df):,}행")

        return df

    def save_results(self, df: pd.DataFrame) -> None:
        """결과 저장"""
        print("\n" + "=" * 70)
        print("[11] 결과 저장")
        print("=" * 70)

        # 컬럼명 변경 및 정렬
        df = df.rename(columns={"배당금": "분배금"})
        save_cols = [
            "기준일자", "코드", "종목명", "자산군",
            "가격", "분배금", "수익률", "무위험수익률", "초과수익률"
        ]
        df["코드"] = df["코드"].astype(str)
        df = df.sort_values(["자산군", "코드", "기준일자"]).reset_index(drop=True)
        df = df[save_cols]

        # 출력 디렉토리 생성
        os.makedirs(self.config.output_dir, exist_ok=True)

        # 저장
        df.to_csv(self.config.output_excess_path, index=False, encoding="utf-8-sig")
        print(f"  저장: {self.config.output_excess_path}")
        print(f"  총 {len(df):,}행")

        return df

    def save_summary(self, df: pd.DataFrame) -> None:
        """요약 통계 저장"""
        print("\n" + "=" * 70)
        print("[12] 요약 통계 저장")
        print("=" * 70)

        summary = self.summary_calculator.compute(df)
        summary.to_excel(self.config.output_summary_path, index=False, engine="openpyxl")

        print(f"  저장: {self.config.output_summary_path}")
        print(f"  총 {len(summary):,}개 상품")

    def print_validation(self, df: pd.DataFrame) -> None:
        """검증 결과 출력"""
        print("\n" + "=" * 70)
        print("[13] 검증")
        print("=" * 70)

        print(f"\n  [기본 통계]")
        print(f"    총 데이터: {len(df):,}행")
        print(f"    상품 수: {df['코드'].nunique():,}개")
        print(f"    거래일 수: {df['기준일자'].nunique():,}일")
        print(f"    기간: {df['기준일자'].min()} ~ {df['기준일자'].max()}")

        print(f"\n  [자산군별 통계]")
        asset_stats = df.groupby("자산군").agg({
            "코드": "nunique",
            "초과수익률": "count"
        })
        asset_stats.columns = ["상품수", "데이터수"]
        print(asset_stats)

        div_cnt = (df["분배금"] > 0).sum()
        print(f"\n  [분배금 반영]")
        print(f"    분배금 반영 건수: {div_cnt:,}건 ({div_cnt / len(df) * 100:.2f}%)")

    def run(self) -> None:
        """전체 파이프라인 실행"""
        # 1. 가격 데이터 로드
        price_data = self.load_price_data()

        # 2. 이상치 날짜 필터링
        price_data = self.filter_anomaly_dates(price_data)

        # 3. 제외 상품 필터링
        price_data = self.filter_excluded_codes(price_data)

        # 4. 배당 데이터 로드
        div_df = self.load_dividend_data()

        # 5. 배당락일 계산
        div_df = self.calculate_ex_dividend_dates(div_df, price_data)

        # 6. 가격 + 배당 병합
        merged_data = self.merge_dividend(price_data, div_df)

        # 7. 수익률 계산
        combined = self.calculate_returns(merged_data)

        # 8. 종목명 통일
        combined = self.unify_product_names(combined)

        # 9. 초과수익률 계산
        combined = self.calculate_excess_returns(combined)

        # 10. 데이터 필터링
        combined = self.filter_data(combined)

        # 11. 결과 저장
        final_df = self.save_results(combined)

        # 12. 요약 통계 저장
        self.save_summary(final_df)

        # 13. 검증
        self.print_validation(final_df)

        print("\n" + "=" * 70)
        print("완료")
        print("=" * 70)


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    config = ReturnConfig()
    pipeline = ReturnCalculationPipeline(config)
    pipeline.run()
