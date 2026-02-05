"""
펀드 분배금 데이터 크롤러
- SEIBRO에서 펀드별분배금지급내역 수집
- ISIN코드로 검색하여 주(좌)당배당액(CASH_ALOC_AMT) 수집
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import time
import os

import config


class FundDividendCrawler:
    """펀드 분배금 데이터 수집기"""

    def __init__(self):
        self.url = "https://seibro.or.kr/websquare/engine/proworks/callServletService.jsp"
        self.headers = {
            "Content-Type": "application/xml; charset=UTF-8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://seibro.or.kr/websquare/control.jsp?w2xPath=/IPORTAL/user/fund/BIP_CNTS05008V.xml&menuNo=152",
            "Origin": "https://seibro.or.kr"
        }
        self.page_size = 100

    def _build_request_xml(self, isin, start_date, end_date, page_num=1):
        """API 요청 XML 생성"""
        return f'''<reqParam action="exerFundByDtrPaySkedulPList" task="ksd.safe.bip.cnts.Fund.process.FundExerPTask">
    <MENU_NO value="152"/>
    <CMM_BTN_ABBR_NM value="total_search,openall,print,hwp,word,pdf,searchIcon,seach,"/>
    <W2XPATH value="/IPORTAL/user/fund/BIP_CNTS05008V.xml"/>
    <ISIN value="{isin}"/>
    <START_DT value="{start_date}"/>
    <END_DT value="{end_date}"/>
    <PAGE_NUM value="{page_num}"/>
    <PAGE_ON_CNT value="{self.page_size}"/>
</reqParam>'''

    def _parse_response(self, xml_text):
        """응답 XML 파싱"""
        records = []

        try:
            root = ET.fromstring(xml_text)

            for data in root.findall('.//data'):
                result = data.find('result')
                if result is not None:
                    record = {}
                    for elem in result:
                        record[elem.tag] = elem.get('value', '')
                    records.append(record)
        except ET.ParseError as e:
            print(f"XML 파싱 에러: {e}")

        return records

    def fetch_by_isin(self, isin, start_date, end_date):
        """
        단일 ISIN의 분배금 내역 조회

        Args:
            isin: ISIN 코드 (예: KRZ501431674)
            start_date: 조회 시작일 (YYYYMMDD)
            end_date: 조회 종료일 (YYYYMMDD)

        Returns:
            분배금 내역 리스트
        """
        payload = self._build_request_xml(isin, start_date, end_date)

        try:
            response = requests.post(
                self.url,
                data=payload.encode('utf-8'),
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            records = self._parse_response(response.text)

            for record in records:
                record['ISIN'] = isin

            return records
        except requests.RequestException as e:
            print(f"요청 에러 (ISIN: {isin}): {e}")
            return []

    def fetch_all_from_isin_list(self, isin_list, start_date=None, end_date=None,
                                  delay=None, save_interval=100):
        """
        ISIN 목록 전체의 분배금 내역 조회

        Args:
            isin_list: ISIN 코드 리스트
            start_date: 조회 시작일 (YYYYMMDD), 기본값: config 설정값
            end_date: 조회 종료일 (YYYYMMDD), 기본값: 오늘
            delay: 요청 간 대기 시간 (초)
            save_interval: 중간 저장 간격 (몇 개마다 저장)

        Returns:
            DataFrame with all fund dividend data
        """
        end_date = end_date or datetime.now().strftime('%Y%m%d')
        start_date = start_date or config.DIVIDEND_START_DATE
        delay = delay or config.API_DELAY

        print(f"=" * 60)
        print(f"SEIBRO 펀드별분배금지급내역 크롤링")
        print(f"조회 기간: {start_date} ~ {end_date}")
        print(f"대상 ISIN 수: {len(isin_list)}개")
        print(f"=" * 60)

        all_records = []
        success_count = 0
        no_data_count = 0

        for idx, isin in enumerate(isin_list, 1):
            records = self.fetch_by_isin(isin, start_date, end_date)

            if records:
                all_records.extend(records)
                success_count += 1
                print(f"[{idx}/{len(isin_list)}] {isin}: {len(records)}건")
            else:
                no_data_count += 1
                print(f"[{idx}/{len(isin_list)}] {isin}: 데이터 없음")

            # 중간 저장
            if idx % save_interval == 0 and all_records:
                temp_df = pd.DataFrame(all_records)
                temp_file = os.path.join(config.OUTPUT_DIR, f"fund_dividend_temp_{idx}.csv")
                temp_df.to_csv(temp_file, index=False, encoding='utf-8-sig')
                print(f"\n>>> 중간 저장: {temp_file} ({len(all_records)}건)\n")

            time.sleep(delay)

        print(f"\n{'=' * 60}")
        print(f"크롤링 완료!")
        print(f"  - 성공 (데이터 있음): {success_count}개")
        print(f"  - 데이터 없음: {no_data_count}개")
        print(f"  - 총 수집 건수: {len(all_records)}건")
        print(f"{'=' * 60}")

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        df = self._process_dataframe(df)

        return df

    def _process_dataframe(self, df):
        """DataFrame 후처리 (컬럼명 변경, 데이터 타입 변환)"""

        column_mapping = {
            'ISIN': 'ISIN코드',
            'ISSUCO_CUSTNO': '발행회사번호',
            'RGT_STD_DT': '기준일자',
            'RGT_RSN_DTAIL_SORT_NM': '배당구분',
            'FIX_TPNM': '배당확정여부',
            'ALOC_WHNM': '현금배당방법',
            'CLERDIV_VAL': '청산분배값',
            'PAY_TERM': '지급기간',
            'SETACC_STDPRC': '결산기준가',
            'SETACC_TAXSTD': '결산과표기준가',
            'CASH_ALOC_AMT': '주좌당배당액',
            'CASH_ALOC_RATIO': '주좌당배당률',
            'TOT_DIV_PAY_AMT': '총분배금',
            'TAX_TPNM': '세금구분',
            'CLER_NOS': '청산차수'
        }

        df = df.rename(columns=column_mapping)

        if '기준일자' in df.columns:
            df['기준일자'] = pd.to_datetime(df['기준일자'], format='%Y%m%d', errors='coerce')

        numeric_cols = ['결산기준가', '결산과표기준가', '주좌당배당액', '주좌당배당률', '총분배금']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        if 'ISIN코드' in df.columns:
            df['펀드코드'] = df['ISIN코드'].str[3:12]

        preferred_order = ['ISIN코드', '펀드코드', '기준일자', '배당구분', '배당확정여부',
                          '현금배당방법', '결산기준가', '결산과표기준가',
                          '주좌당배당액', '주좌당배당률', '총분배금',
                          '지급기간', '세금구분', '청산차수', '발행회사번호']

        existing_cols = [col for col in preferred_order if col in df.columns]
        other_cols = [col for col in df.columns if col not in preferred_order]
        df = df[existing_cols + other_cols]

        return df


def run():
    """펀드 분배금 크롤러 실행"""
    print("=" * 60)
    print("펀드 분배금 데이터 크롤러")
    print("=" * 60)

    # ISIN 목록 로드
    print("ISIN 목록 로딩...")
    try:
        df_input = pd.read_excel(config.FUND_ISIN_LIST_FILE)
    except Exception as e:
        print(f"펀드 ISIN 목록 파일 읽기 오류: {e}")
        return None

    # 유효한 ISIN만 추출 (KRZ로 시작, '-' 제외)
    isin_col = df_input['ISIN코드']
    valid_isin = isin_col[(isin_col != '-') & (isin_col.notna()) & (isin_col.str.startswith('KRZ', na=False))]
    isin_list = valid_isin.tolist()

    print(f"유효한 ISIN 수: {len(isin_list)}개")

    # 크롤링 실행
    crawler = FundDividendCrawler()

    df = crawler.fetch_all_from_isin_list(
        isin_list=isin_list,
        start_date=config.DIVIDEND_START_DATE,
        end_date=datetime.now().strftime('%Y%m%d'),
        save_interval=100
    )

    if df.empty:
        print("수집된 데이터가 없습니다.")
        return None

    # 원본 데이터와 merge (상품 코드, 단축코드 추가)
    if '상품 코드' in df_input.columns and '단축코드' in df_input.columns:
        df_input_subset = df_input[['ISIN코드', '상품 코드', '단축코드']].drop_duplicates()
        df = df.merge(df_input_subset, on='ISIN코드', how='left')
        print(f"원본 데이터와 merge 완료 (상품 코드, 단축코드 추가)")

    # 결과 저장
    output_file = os.path.join(config.OUTPUT_DIR, f"fund_dividend_{datetime.now().strftime('%Y%m%d')}.xlsx")
    df.to_excel(output_file, index=False)
    print(f"\n저장 완료: {output_file}")

    # 요약 출력
    print(f"\n컬럼: {df.columns.tolist()}")
    print(f"\n기준일자 범위: {df['기준일자'].min()} ~ {df['기준일자'].max()}")
    print(f"\n배당구분별 건수:")
    print(df['배당구분'].value_counts())

    print(f"\n샘플 데이터 (주좌당배당액 포함):")
    print(df[['ISIN코드', '기준일자', '결산기준가', '주좌당배당액', '주좌당배당률', '총분배금']].head(10))

    return df


if __name__ == "__main__":
    run()
