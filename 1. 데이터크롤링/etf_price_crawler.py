"""
ETF 가격 데이터 크롤러
- KRX API를 통해 ETF 일별 시세 데이터 수집
- 수집 후 특정 기준일에 데이터가 있는 ETF만 필터링
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os
import glob

import config


class ETFPriceCollector:
    """ETF 가격 데이터 수집기"""

    def __init__(self, api_key=None):
        self.api_key = api_key or config.KRX_API_KEY
        self.base_url = "https://data-dbg.krx.co.kr/svc/apis/etp/etf_bydd_trd"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

    def fetch_etf_data(self, date):
        """특정 날짜의 ETF 데이터 조회"""
        payload = {"basDd": date}

        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                if 'OutBlock_1' in data and data['OutBlock_1']:
                    return data['OutBlock_1']
                else:
                    return []
            else:
                print(f"API 오류 ({date}): Status {response.status_code}")
                return []

        except requests.exceptions.RequestException as e:
            print(f"요청 오류 ({date}): {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"JSON 파싱 오류 ({date}): {e}")
            return []

    def get_date_range(self, total_years):
        """수집할 날짜 범위 생성 (모든 날짜)"""
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=365 * total_years)

        dates = []
        current_date = start_date
        while current_date <= end_date:
            dates.append(current_date.strftime('%Y%m%d'))
            current_date += timedelta(days=1)

        return dates

    def collect_data(self, total_years=None, delay=None):
        """
        ETF 가격 데이터 수집 (년 단위로 나누어 저장)

        Args:
            total_years: 수집할 기간 (년), 기본값: config 설정값
            delay: API 호출 간격 (초), 기본값: config 설정값

        Returns:
            생성된 파일 목록
        """
        total_years = total_years or config.PRICE_CRAWL_YEARS
        delay = delay or config.API_DELAY

        end_date = datetime.now() - timedelta(days=1)

        print(f"ETF {total_years}년치 데이터를 1년씩 나누어 수집합니다.")
        print(f"종료일: {end_date.strftime('%Y-%m-%d')}")
        print("=" * 60)

        collected_files = []

        for year_idx in range(total_years):
            year_end = end_date - timedelta(days=365 * year_idx)
            year_start = year_end - timedelta(days=364)

            # 이미 파일이 있으면 스킵
            filename = f"etf_price_{year_start.strftime('%Y')}.csv"
            filepath = os.path.join(config.OUTPUT_DIR, filename)

            if os.path.exists(filepath):
                print(f"\n[{year_idx + 1}/{total_years}년차] 이미 존재 - 스킵: {filename}")
                collected_files.append(filepath)
                continue

            print(f"\n[{year_idx + 1}/{total_years}년차] 수집 시작")
            print(f"기간: {year_start.strftime('%Y-%m-%d')} ~ {year_end.strftime('%Y-%m-%d')}")

            year_data = self._collect_year_data(year_start, year_end, year_idx + 1, total_years, delay)

            if year_data:
                filename = f"etf_price_{year_start.strftime('%Y')}.csv"
                filepath = os.path.join(config.OUTPUT_DIR, filename)

                df = self._save_to_csv(year_data, filepath)
                collected_files.append(filepath)

                print(f"[완료] {filename} 저장 (레코드 수: {len(year_data):,})")
            else:
                print(f"[실패] {year_start.strftime('%Y')}년 데이터 수집 실패")

        print("\n" + "=" * 60)
        print("전체 수집 완료!")
        print(f"생성된 파일: {len(collected_files)}개")

        return collected_files

    def _collect_year_data(self, start_date, end_date, current_year, total_years, delay):
        """1년치 데이터 수집"""
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.strftime('%Y%m%d'))
            current += timedelta(days=1)

        print(f"  총 {len(dates)}일을 처리합니다.")

        all_data = []
        successful_dates = 0
        progress_interval = max(1, len(dates) // 12)

        for i, date in enumerate(dates, 1):
            data = self.fetch_etf_data(date)
            if data:
                all_data.extend(data)
                successful_dates += 1

            if i % progress_interval == 0 or i == len(dates):
                progress_pct = (i / len(dates)) * 100
                print(f"  [{current_year}/{total_years}년차] {progress_pct:.1f}% - {date} | 데이터: {len(all_data):,}개")

            if delay > 0:
                time.sleep(delay)

        print(f"  완료: 데이터 있는 날짜 {successful_dates}/{len(dates)}일, 총 레코드 {len(all_data):,}개")

        return all_data

    def _save_to_csv(self, data, filepath):
        """데이터를 CSV 파일로 저장"""
        if not data:
            return None

        df = pd.DataFrame(data)

        column_mapping = {
            'BAS_DD': '기준일자',
            'ISU_CD': '종목코드',
            'ISU_NM': '종목명',
            'TDD_CLSPRC': '종가',
            'CMPPREVDD_PRC': '대비',
            'FLUC_RT': '등락률',
            'NAV': '순자산가치',
            'TDD_OPNPRC': '시가',
            'TDD_HGPRC': '고가',
            'TDD_LWPRC': '저가',
            'ACC_TRDVOL': '거래량',
            'ACC_TRDVAL': '거래대금',
            'MKTCAP': '시가총액',
            'INVSTASST_NETASST_TOTAMT': '순자산총액',
            'LIST_SHRS': '상장좌수',
            'IDX_IND_NM': '기초지수명',
            'OBJ_STKPRC_IDX': '기초지수종가',
            'CMPPREVDD_IDX': '기초지수대비',
            'FLUC_RT_IDX': '기초지수등락률'
        }

        df = df.rename(columns=column_mapping)
        df['기준일자'] = pd.to_datetime(df['기준일자'], format='%Y%m%d')
        df = df.sort_values(['기준일자', '종목코드'])

        df.to_csv(filepath, index=False, encoding='utf-8-sig')

        return df


def filter_etf_data(reference_date=None):
    """
    수집된 ETF 데이터 필터링
    - 특정 기준일에 데이터가 있는 ETF만 선별
    - 모든 연도 데이터를 통합하여 저장

    Args:
        reference_date: 기준일 (datetime), 기본값: 가장 최근 데이터의 마지막 거래일

    Returns:
        통합된 DataFrame
    """
    print("=" * 60)
    print("ETF 데이터 필터링 시작")
    print("=" * 60)

    # ETF 코드 목록 로드
    try:
        print("\nETF 코드 파일 읽는 중...")
        etf_codes_df = pd.read_excel(config.ETF_CODE_LIST_FILE, dtype={'코드': str})
        etf_codes_df['코드'] = etf_codes_df['코드'].str.replace('.0', '', regex=False)
        print(f"  - ETF 코드 데이터 로드 완료: {len(etf_codes_df)}개")
    except Exception as e:
        print(f"ETF 코드 파일 읽기 오류: {e}")
        return None

    # 수집된 CSV 파일 목록
    csv_files = glob.glob(os.path.join(config.OUTPUT_DIR, "etf_price_*.csv"))
    csv_files.sort()

    if not csv_files:
        print("수집된 ETF 가격 파일이 없습니다.")
        return None

    print(f"\n발견된 CSV 파일: {len(csv_files)}개")

    # 가장 최근 파일에서 기준일 결정
    if reference_date is None:
        latest_file = max(csv_files)
        df_latest = pd.read_csv(latest_file, encoding='utf-8-sig', dtype={'종목코드': str})
        df_latest['기준일자'] = pd.to_datetime(df_latest['기준일자'])
        reference_date = df_latest['기준일자'].max()

    print(f"\n기준일: {reference_date.strftime('%Y-%m-%d')}")

    # 기준일에 데이터가 있는 ETF 코드 추출
    selected_etf_codes = []

    for csv_file in csv_files:
        df = pd.read_csv(csv_file, encoding='utf-8-sig', dtype={'종목코드': str})
        df['기준일자'] = pd.to_datetime(df['기준일자'])
        df['종목코드'] = df['종목코드'].str.replace('.0', '', regex=False)

        ref_data = df[df['기준일자'] == reference_date]
        codes_in_ref = ref_data['종목코드'].unique().tolist()

        for code in codes_in_ref:
            if code in etf_codes_df['코드'].values and code not in selected_etf_codes:
                selected_etf_codes.append(code)

    print(f"\n기준일에 데이터가 있는 ETF: {len(selected_etf_codes)}개")

    if not selected_etf_codes:
        print("기준일에 데이터가 있는 ETF가 없습니다.")
        return None

    # 선택된 ETF 데이터만 수집
    all_etf_data = []

    for csv_file in csv_files:
        print(f"  처리 중: {os.path.basename(csv_file)}")
        df = pd.read_csv(csv_file, encoding='utf-8-sig', dtype={'종목코드': str})
        df['기준일자'] = pd.to_datetime(df['기준일자'])
        df['종목코드'] = df['종목코드'].str.replace('.0', '', regex=False)

        filtered_df = df[df['종목코드'].isin(selected_etf_codes)]

        if len(filtered_df) > 0:
            all_etf_data.append(filtered_df)
            print(f"    {len(filtered_df):,}행 추가")

    # 데이터 통합 및 저장
    if not all_etf_data:
        print("수집된 데이터가 없습니다.")
        return None

    print(f"\n데이터 통합 중...")
    combined_df = pd.concat(all_etf_data, ignore_index=True)
    sorted_df = combined_df.sort_values(['종목코드', '기준일자'], ascending=[True, False])

    output_file = os.path.join(config.OUTPUT_DIR, "etf_price_filtered.csv")
    sorted_df.to_csv(output_file, index=False, encoding='utf-8-sig')

    print(f"\n" + "=" * 60)
    print("결과 요약")
    print("=" * 60)
    print(f"선택된 ETF 개수: {len(selected_etf_codes)}개")
    print(f"총 데이터 행 수: {len(sorted_df):,}행")
    print(f"데이터 기간: {sorted_df['기준일자'].min().strftime('%Y-%m-%d')} ~ {sorted_df['기준일자'].max().strftime('%Y-%m-%d')}")
    print(f"저장 파일: {output_file}")

    return sorted_df


def run():
    """ETF 가격 크롤러 실행"""
    print("=" * 60)
    print("ETF 가격 데이터 크롤러")
    print("=" * 60)

    collector = ETFPriceCollector()

    # 1. 데이터 수집
    collected_files = collector.collect_data()

    if not collected_files:
        print("수집된 데이터가 없습니다.")
        return None

    # 2. 데이터 필터링
    df = filter_etf_data()

    return df


if __name__ == "__main__":
    run()
