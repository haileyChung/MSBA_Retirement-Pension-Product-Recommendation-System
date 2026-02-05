"""
리츠 가격 데이터 크롤러
- KRX API를 통해 유가증권 일별 시세 데이터 수집
- 리츠 코드 목록에 해당하는 종목만 필터링
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os

import config


class ReitsPriceCollector:
    """리츠 가격 데이터 수집기"""

    def __init__(self, api_key=None):
        self.api_key = api_key or config.KRX_API_KEY
        self.base_url = "https://data-dbg.krx.co.kr/svc/apis/sto/stk_bydd_trd"
        self.headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        self.reits_codes = self._load_reits_codes()

    def _load_reits_codes(self):
        """통합 리츠 목록에서 리츠 코드를 로드"""
        try:
            df = pd.read_excel(config.REITS_LIST_FILE)
            reits_info = {}

            for _, row in df.iterrows():
                code = str(row['리츠 코드']).zfill(6)
                name = row['리츠명']
                feature = row['특징'] if '특징' in df.columns else '거래 중'
                reits_info[code] = {'name': name, 'feature': feature}

            print(f"리츠 정보 {len(reits_info)}개를 로드했습니다.")
            for code, info in reits_info.items():
                print(f"  - {info['name']} ({code}) - 특징: {info['feature']}")

            return reits_info

        except Exception as e:
            print(f"리츠 정보 파일 로드 오류: {e}")
            return {}

    def fetch_all_stock_data(self, date):
        """특정 날짜의 모든 유가증권 데이터 조회"""
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

    def collect_data(self, max_years=None, delay=None):
        """
        리츠 가격 데이터 수집

        Args:
            max_years: 수집할 기간 (년), 기본값: config 설정값
            delay: API 호출 간격 (초), 기본값: config 설정값

        Returns:
            (리츠 데이터 리스트, 실패 리츠 리스트)
        """
        max_years = max_years or config.PRICE_CRAWL_YEARS
        delay = delay or config.API_DELAY

        if not self.reits_codes:
            print("리츠 정보를 로드할 수 없습니다.")
            return None, None

        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=365 * max_years)

        print(f"리츠 {max_years}년치 전체 유가증권 데이터 수집 시작")
        print(f"수집 기간: {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}")
        print(f"대상 리츠: {len(self.reits_codes)}개")
        print("=" * 60)

        # 날짜 범위 생성
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current.strftime('%Y%m%d'))
            current += timedelta(days=1)

        print(f"총 {len(dates)}일을 처리합니다.")
        print("=" * 60)

        all_data = []
        successful_dates = 0
        start_time = datetime.now()
        progress_interval = max(1, len(dates) // 50)

        for i, date in enumerate(dates, 1):
            daily_data = self.fetch_all_stock_data(date)

            if daily_data:
                all_data.extend(daily_data)
                successful_dates += 1

            if i % progress_interval == 0 or i == len(dates):
                progress_pct = (i / len(dates)) * 100
                elapsed = datetime.now() - start_time
                remaining_time = elapsed * (len(dates) - i) / i if i > 0 else timedelta(0)

                print(f"진행률: {progress_pct:.1f}% ({i}/{len(dates)}) | "
                      f"날짜: {date} | 누적 데이터: {len(all_data):,}개 | "
                      f"남은 시간: {str(remaining_time).split('.')[0]}")

            if delay > 0:
                time.sleep(delay)

        total_time = datetime.now() - start_time
        print(f"\n전체 데이터 수집 완료!")
        print(f"소요 시간: {str(total_time).split('.')[0]}")
        print(f"수집된 총 레코드: {len(all_data):,}개")
        print(f"성공한 거래일: {successful_dates}/{len(dates)}일")

        # 리츠 데이터 필터링
        print("\n" + "=" * 60)
        print("리츠 데이터 필터링 중...")

        reits_data, failed_reits = self._filter_reits_data(all_data)

        return reits_data, failed_reits

    def _filter_reits_data(self, all_data):
        """수집된 데이터에서 리츠만 필터링"""
        latest_data_check = {}

        for code in self.reits_codes.keys():
            code_data = [item for item in all_data if item.get('ISU_CD') == code]
            if code_data:
                code_data.sort(key=lambda x: x.get('BAS_DD', ''), reverse=True)
                latest_date = code_data[0].get('BAS_DD', '')
                latest_data_check[code] = {
                    'latest_date': latest_date,
                    'data_count': len(code_data)
                }

        reits_data = []
        found_codes = set()
        failed_reits = []

        for code, info in self.reits_codes.items():
            if info['feature'] == '거래 중':
                if code in latest_data_check:
                    check_info = latest_data_check[code]
                    code_data = [item for item in all_data if item.get('ISU_CD') == code]
                    reits_data.extend(code_data)
                    found_codes.add(code)
                    print(f"  [성공] {info['name']} ({code}): {check_info['data_count']:,}개 데이터 (최근: {check_info['latest_date']})")
                else:
                    failed_reits.append({
                        'name': info['name'],
                        'code': code,
                        'reason': '수집 기간 내 거래 데이터 없음'
                    })
                    print(f"  [실패] {info['name']} ({code}): 데이터 없음")
            else:
                failed_reits.append({
                    'name': info['name'],
                    'code': code,
                    'reason': f"특징이 '거래 중'이 아님 (실제: {info['feature']})"
                })
                print(f"  [제외] {info['name']} ({code}): 거래 중이 아님 ({info['feature']})")

        print(f"\n필터링 완료!")
        print(f"선택된 리츠 데이터: {len(reits_data):,}개")
        print(f"선택된 리츠: {len(found_codes)}/{len(self.reits_codes)}개")
        print(f"제외된 리츠: {len(failed_reits)}개")

        return reits_data, failed_reits

    def save_to_csv(self, data, failed_reits=None):
        """데이터를 CSV 파일로 저장"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if data:
            df = pd.DataFrame(data)

            column_mapping = {
                'BAS_DD': '기준일자',
                'ISU_CD': '종목코드',
                'ISU_NM': '종목명',
                'MKT_NM': '시장구분',
                'SECT_TP_NM': '소속부',
                'TDD_CLSPRC': '종가',
                'CMPPREVDD_PRC': '대비',
                'FLUC_RT': '등락률',
                'TDD_OPNPRC': '시가',
                'TDD_HGPRC': '고가',
                'TDD_LWPRC': '저가',
                'ACC_TRDVOL': '거래량',
                'ACC_TRDVAL': '거래대금',
                'MKTCAP': '시가총액',
                'LIST_SHRS': '상장주식수'
            }

            df = df.rename(columns=column_mapping)
            df['기준일자'] = pd.to_datetime(df['기준일자'], format='%Y%m%d')
            df = df.sort_values(['기준일자', '종목코드'])

            main_filename = os.path.join(config.OUTPUT_DIR, f"reits_price_{timestamp}.csv")
            df.to_csv(main_filename, index=False, encoding='utf-8-sig')
            print(f"\n메인 데이터 저장: {main_filename}")

            print(f"\n" + "=" * 60)
            print(f"최종 데이터 요약")
            print(f"=" * 60)
            print(f"전체 레코드 수: {len(df):,}")
            print(f"고유 종목 수: {df['종목코드'].nunique()}")
            print(f"수집 기간: {df['기준일자'].min().strftime('%Y-%m-%d')} ~ {df['기준일자'].max().strftime('%Y-%m-%d')}")
            print(f"수집 일수: {df['기준일자'].nunique()}일")

            print(f"\n종목별 데이터 수:")
            print("-" * 60)
            stock_summary = df.groupby(['종목코드', '종목명']).agg({
                '기준일자': ['count', 'min', 'max']
            }).round(0)
            stock_summary.columns = ['데이터수', '시작일', '종료일']
            stock_summary = stock_summary.reset_index()
            stock_summary = stock_summary.sort_values('데이터수', ascending=False)

            for _, row in stock_summary.iterrows():
                print(f"{row['종목명']} ({row['종목코드']}): {int(row['데이터수']):,}개 "
                      f"({row['시작일'].strftime('%Y-%m-%d')} ~ {row['종료일'].strftime('%Y-%m-%d')})")

        if failed_reits:
            failed_df = pd.DataFrame(failed_reits)
            failed_filename = os.path.join(config.OUTPUT_DIR, f"reits_failed_list_{timestamp}.csv")
            failed_df.to_csv(failed_filename, index=False, encoding='utf-8-sig')
            print(f"\n실패 목록 저장: {failed_filename}")


def run():
    """리츠 가격 크롤러 실행"""
    print("=" * 60)
    print("리츠 가격 데이터 크롤러")
    print("=" * 60)

    collector = ReitsPriceCollector()

    if not collector.reits_codes:
        print("리츠 정보를 로드할 수 없습니다. 파일 경로를 확인해주세요.")
        return None

    reits_data, failed_reits = collector.collect_data()
    collector.save_to_csv(reits_data, failed_reits)

    return reits_data


if __name__ == "__main__":
    run()
