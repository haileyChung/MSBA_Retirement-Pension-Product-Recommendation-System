"""
펀드 가격 데이터 크롤러
- KOFIA API를 통해 펀드 기준가격 데이터 수집
- 배치 단위로 수집하여 중간 저장
- 수집 완료 후 모든 CSV 파일 통합
"""

import requests
import pandas as pd
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import time
import random
import os
import glob

import config


class FundPriceCollector:
    """펀드 가격 데이터 수집기"""

    def __init__(self):
        self.url = "https://dis.kofia.or.kr/proframeWeb/XMLSERVICES"
        self.headers = {
            'Content-Type': 'text/xml; charset=UTF-8',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def _get_fund_data(self, fund_code, start_date, end_date, period_name):
        """펀드코드로 기준가격 데이터 가져오기"""
        print(f"      {period_name} 데이터 수집 중... ({start_date}~{end_date})")

        xml_data = f"""<?xml version="1.0" encoding="utf-8"?>
<message>
    <proframeHeader>
        <pfmAppName>FS-DIS2</pfmAppName>
        <pfmSvcName>DISFundStdPrcStutSO</pfmSvcName>
        <pfmFnName>select</pfmFnName>
    </proframeHeader>
    <systemHeader></systemHeader>
    <DISCondFuncDTO>
        <tmpV30>{start_date}</tmpV30>
        <tmpV31>{end_date}</tmpV31>
        <tmpV10>0</tmpV10>
        <tmpV12>{fund_code}</tmpV12>
    </DISCondFuncDTO>
</message>"""

        try:
            response = requests.post(self.url, data=xml_data, headers=self.headers, timeout=30)
            if response.status_code == 200:
                root = ET.fromstring(response.text)
                data_list = []

                for select_meta in root.findall('.//selectMeta'):
                    row = {}
                    for child in select_meta:
                        if child.tag in ['tmpV1', 'tmpV2', 'tmpV3', 'tmpV4', 'tmpV5', 'tmpV6', 'tmpV7', 'tmpV8', 'tmpV9', 'tmpV10', 'tmpV12'] and child.text:
                            row[child.tag] = child.text.strip()
                    if len(row) >= 3:
                        data_list.append(row)

                if data_list:
                    df = pd.DataFrame(data_list)
                    df = df.rename(columns={
                        'tmpV1': '기준일자',
                        'tmpV2': '기준가격',
                        'tmpV3': '전일대비등락(원)',
                        'tmpV4': '과표기준가격(원)',
                        'tmpV5': '설정원본(백만원)',
                        'tmpV6': 'KOSPI(%)',
                        'tmpV7': '국채 주요금리(%)',
                        'tmpV8': '회사채 주요금리(%)',
                        'tmpV9': 'CALL 주요금리(%)',
                        'tmpV10': 'CP 주요금리(%)',
                        'tmpV12': '표준코드'
                    })
                    print(f"      [성공] {period_name}: {len(df)}개 데이터")
                    return df
                else:
                    print(f"      [실패] {period_name}: 데이터 없음")
            else:
                print(f"      [실패] {period_name}: 서버 오류 {response.status_code}")
        except Exception as e:
            print(f"      [실패] {period_name}: {str(e)[:30]}")

        return None

    def _get_multi_year_data(self, company_name, fund_name, fund_code, total_years=None):
        """여러 년치 데이터 수집 (3년씩 나누어서)"""
        total_years = total_years or config.PRICE_CRAWL_YEARS
        print(f"  {total_years}년치 데이터 수집 시작 - 펀드코드: {fund_code}")

        end_of_period = datetime.now() - timedelta(days=1)
        all_data = []

        # 3년 단위로 나누어 수집
        periods = []
        for i in range(0, total_years, 3):
            start_years = i
            end_years = min(i + 3, total_years)
            period_name = f"{start_years}-{end_years}년 전" if start_years > 0 else f"최근 {end_years}년"
            periods.append((period_name, start_years, end_years))

        for period_name, start_years, end_years in periods:
            start_dt = end_of_period - timedelta(days=end_years * 365)
            end_dt = end_of_period - timedelta(days=start_years * 365)

            start_date = start_dt.strftime('%Y%m%d')
            end_date = end_dt.strftime('%Y%m%d')

            df = self._get_fund_data(fund_code, start_date, end_date, period_name)
            if df is not None and len(df) > 0:
                df['운용사'] = company_name
                df['펀드명'] = fund_name
                df['펀드코드'] = fund_code
                all_data.append(df)

            wait_time = random.uniform(1.0, 3.0)
            print(f"      {wait_time:.1f}초 대기...")
            time.sleep(wait_time)

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            print(f"  [완료] 총 {len(result)}개 데이터 수집!")
            return result
        else:
            print(f"  [실패] 모든 기간에서 데이터 없음")
            return None

    def _read_fund_list(self):
        """엑셀 파일에서 펀드 목록 읽기"""
        try:
            print(f"펀드 목록 파일 읽는 중: {config.FUND_CODE_LIST_FILE}")

            df = pd.read_excel(config.FUND_CODE_LIST_FILE, sheet_name=0)
            print(f"파일 크기: {df.shape}")

            fund_list = []
            for idx, row in df.iterrows():
                fund_name = str(row.iloc[0]).strip()
                company_name = str(row.iloc[3]).strip()
                fund_code = str(row.iloc[1]).strip()

                if all(x != 'nan' and len(x) > 2 for x in [fund_name, company_name, fund_code]):
                    fund_list.append({
                        'fund_name': fund_name,
                        'company': company_name,
                        'fund_code': fund_code
                    })

            print(f"총 {len(fund_list)}개 펀드 발견")
            return fund_list

        except Exception as e:
            print(f"엑셀 파일 읽기 오류: {e}")
            return []

    def _process_batch(self, fund_list, start_num, batch_size=None):
        """배치 처리"""
        batch_size = batch_size or config.FUND_BATCH_SIZE

        end_num = min(start_num + batch_size, len(fund_list))
        batch_funds = fund_list[start_num - 1:end_num]
        batch_num = (start_num - 1) // batch_size + 1
        total_batches = (len(fund_list) + batch_size - 1) // batch_size

        print(f"\n" + "=" * 60)
        print(f"배치 {batch_num}/{total_batches} 처리 시작!")
        print(f"처리 범위: {start_num}~{end_num}번째 펀드 ({len(batch_funds)}개)")
        print(f"전체 진행률: {((batch_num - 1) / total_batches * 100):.1f}%")
        print("=" * 60)

        success_data = []
        failed_funds = []

        for i, fund_info in enumerate(batch_funds, 1):
            current_num = start_num + i - 1

            print(f"\n[{current_num}/{len(fund_list)}] 펀드 처리 중... (배치 내 {i}/{len(batch_funds)})")
            print(f"운용사: {fund_info['company']}")
            print(f"펀드명: {fund_info['fund_name'][:60]}...")
            print(f"펀드코드: {fund_info['fund_code']}")

            try:
                data = self._get_multi_year_data(fund_info['company'], fund_info['fund_name'], fund_info['fund_code'])
                if data is not None:
                    success_data.append(data)
                    print(f"[성공] 펀드 처리 완료!")
                else:
                    failed_funds.append({
                        'company': fund_info['company'],
                        'fund_name': fund_info['fund_name'],
                        'fund_code': fund_info['fund_code'],
                        'reason': '데이터 없음'
                    })
                    print(f"[실패] 펀드 처리 실패!")
            except Exception as e:
                failed_funds.append({
                    'company': fund_info['company'],
                    'fund_name': fund_info['fund_name'],
                    'fund_code': fund_info['fund_code'],
                    'reason': str(e)[:50]
                })
                print(f"[오류] 펀드 처리 오류: {str(e)[:50]}")

            batch_progress = (i / len(batch_funds)) * 100
            print(f"배치 진행률: {batch_progress:.1f}% ({i}/{len(batch_funds)})")

            if i < len(batch_funds):
                wait_time = random.uniform(1.0, 3.0)
                print(f"{wait_time:.1f}초 대기...")
                time.sleep(wait_time)

        # 결과 저장
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')

        print(f"\n배치 {batch_num} 결과 저장 중...")

        if success_data:
            success_df = pd.concat(success_data, ignore_index=True)
            success_file = os.path.join(config.OUTPUT_DIR, f"fund_price_batch{batch_num}_{timestamp}.csv")
            success_df.to_csv(success_file, index=False, encoding='utf-8-sig')

            print(f"[저장] 성공 데이터: {success_file}")
            print(f"  데이터 개수: {len(success_df):,}행")
            print(f"  성공 펀드: {len(success_data)}개")

        if failed_funds:
            failed_df = pd.DataFrame(failed_funds)
            failed_file = os.path.join(config.OUTPUT_DIR, f"fund_failed_batch{batch_num}_{timestamp}.csv")
            failed_df.to_csv(failed_file, index=False, encoding='utf-8-sig')

            print(f"[저장] 실패 목록: {failed_file}")
            print(f"  실패 펀드: {len(failed_funds)}개")

        print(f"\n배치 {batch_num} 완료!")
        print(f"[성공] {len(success_data)}개 | [실패] {len(failed_funds)}개")
        if len(success_data) + len(failed_funds) > 0:
            print(f"성공률: {len(success_data) / (len(success_data) + len(failed_funds)) * 100:.1f}%")

    def collect_data(self, start_num=1, batch_size=None):
        """
        펀드 가격 데이터 수집 (배치 단위)

        Args:
            start_num: 시작할 펀드 번호 (1부터 시작)
            batch_size: 배치 크기, 기본값: config 설정값
        """
        batch_size = batch_size or config.FUND_BATCH_SIZE
        batch_delay = config.FUND_BATCH_DELAY

        fund_list = self._read_fund_list()
        if not fund_list:
            return

        print(f"\n총 {len(fund_list)}개 펀드 발견")
        print(f"{start_num}번째 펀드부터 {batch_size}개씩 처리합니다.")
        print("전체 처리 시작!")

        current_start = start_num
        while current_start <= len(fund_list):
            self._process_batch(fund_list, current_start, batch_size)
            current_start += batch_size

            if current_start <= len(fund_list):
                print(f"\n{batch_delay}초 후 다음 배치 시작...")
                time.sleep(batch_delay)

        print("\n모든 처리 완료!")


def merge_fund_csv_files():
    """
    수집된 펀드 CSV 파일들을 하나로 통합

    Returns:
        통합된 파일 경로
    """
    print("=" * 60)
    print("펀드 데이터 통합 시작")
    print("=" * 60)

    csv_files = glob.glob(os.path.join(config.OUTPUT_DIR, "fund_price_batch*.csv"))

    if not csv_files:
        print("통합할 펀드 가격 파일이 없습니다.")
        return None

    all_data = []

    print(f"발견된 CSV 파일: {len(csv_files)}개")

    for file in csv_files:
        print(f"  처리 중: {os.path.basename(file)}")

        try:
            df = pd.read_csv(file, encoding='utf-8-sig')
            df = df.dropna(how='all')
            df.columns = df.columns.str.strip()
            all_data.append(df)
            print(f"    {len(df)}행 데이터 추가됨")

        except Exception as e:
            print(f"    오류 발생: {e}")
            continue

    if not all_data:
        print("처리할 데이터가 없습니다.")
        return None

    print("\n데이터 통합 중...")
    combined_df = pd.concat(all_data, ignore_index=True, sort=False)

    output_filename = os.path.join(config.OUTPUT_DIR, f"fund_price_merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    try:
        combined_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n[완료] 통합 완료!")
        print(f"  출력 파일: {output_filename}")
        print(f"  총 행 수: {len(combined_df):,}행")
        print(f"  총 컬럼 수: {len(combined_df.columns)}개")

        print(f"\n컬럼 목록:")
        for i, col in enumerate(combined_df.columns, 1):
            print(f"  {i:2d}. {col}")

        print(f"\n기본 정보:")
        print(f"  처리된 파일 수: {len(all_data)}개")

    except Exception as e:
        print(f"CSV 저장 실패: {e}")
        return None

    return output_filename


def run(start_num=1):
    """
    펀드 가격 크롤러 실행

    Args:
        start_num: 시작할 펀드 번호 (1부터 시작)
    """
    print("=" * 60)
    print("펀드 가격 데이터 크롤러")
    print("=" * 60)

    collector = FundPriceCollector()

    # 1. 데이터 수집
    collector.collect_data(start_num=start_num)

    # 2. CSV 파일 통합
    merge_fund_csv_files()


if __name__ == "__main__":
    # 1번 펀드부터 시작 (중단된 경우 해당 번호로 수정)
    run(start_num=1)
