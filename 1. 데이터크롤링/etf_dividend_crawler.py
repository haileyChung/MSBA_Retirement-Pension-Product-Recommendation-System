"""
ETF 분배금 데이터 크롤러
- SEIBRO에서 ETF 분배금지급현황 수집
- 날짜 범위로 전체 조회 (별도 입력 파일 불필요)
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import time
import os

import config


class ETFDividendCrawler:
    """ETF 분배금 데이터 수집기"""

    def __init__(self):
        self.url = "https://seibro.or.kr/websquare/engine/proworks/callServletService.jsp"
        self.headers = {
            "Content-Type": "application/xml; charset=UTF-8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://seibro.or.kr/websquare/control.jsp?w2xPath=/IPORTAL/user/etf/BIP_CNTS06030V.xml&menuNo=179",
            "Origin": "https://seibro.or.kr"
        }
        self.page_size = 30

    def _build_request_xml(self, start_page, end_page, from_date, to_date):
        """API 요청 XML 생성"""
        return f'''<reqParam action="exerInfoDtramtPayStatPlist" task="ksd.safe.bip.cnts.etf.process.EtfExerInfoPTask">
    <MENU_NO value="179"/>
    <CMM_BTN_ABBR_NM value="total_search,openall,print,hwp,word,pdf,searchIcon,searchIcon,seach,searchIcon,seach,"/>
    <W2XPATH value="/IPORTAL/user/etf/BIP_CNTS06030V.xml"/>
    <etf_sort_level_cd value="0"/>
    <etf_big_sort_cd value=""/>
    <START_PAGE value="{start_page}"/>
    <END_PAGE value="{end_page}"/>
    <etf_sort_cd value=""/>
    <isin value=""/>
    <mngco_custno value=""/>
    <RGT_RSN_DTAIL_SORT_CD value=""/>
    <fromRGT_STD_DT value="{from_date}"/>
    <toRGT_STD_DT value="{to_date}"/>
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

    def fetch_page(self, page_num, from_date, to_date):
        """단일 페이지 데이터 조회"""
        start_page = (page_num - 1) * self.page_size + 1
        end_page = page_num * self.page_size

        payload = self._build_request_xml(start_page, end_page, from_date, to_date)

        try:
            response = requests.post(
                self.url,
                data=payload.encode('utf-8'),
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return self._parse_response(response.text)
        except requests.RequestException as e:
            print(f"요청 에러 (페이지 {page_num}): {e}")
            return []

    def fetch_all(self, from_date=None, to_date=None, max_pages=200, delay=None):
        """
        전체 데이터 조회

        Args:
            from_date: 조회 시작일 (YYYYMMDD), 기본값: config 설정값
            to_date: 조회 종료일 (YYYYMMDD), 기본값: 오늘
            max_pages: 최대 조회 페이지 수
            delay: 요청 간 대기 시간 (초), 기본값: config 설정값

        Returns:
            DataFrame with ETF dividend data
        """
        to_date = to_date or datetime.now().strftime('%Y%m%d')
        from_date = from_date or config.DIVIDEND_START_DATE
        delay = delay or config.API_DELAY

        all_records = []
        page_num = 1

        print(f"조회 기간: {from_date} ~ {to_date}")
        print("데이터 수집 시작...")

        while page_num <= max_pages:
            records = self.fetch_page(page_num, from_date, to_date)

            if not records:
                print(f"페이지 {page_num}: 데이터 없음 - 수집 완료")
                break

            all_records.extend(records)
            print(f"페이지 {page_num}: {len(records)}건 수집 (누적: {len(all_records)}건)")

            if len(records) < self.page_size:
                print("마지막 페이지 도달 - 수집 완료")
                break

            page_num += 1
            time.sleep(delay)

        if not all_records:
            return pd.DataFrame()

        df = pd.DataFrame(all_records)

        # 컬럼명 한글화
        column_mapping = {
            'ISIN': 'ISIN코드',
            'KOR_SECN_NM': '종목명',
            'ETF_SORT_NM': '유형',
            'REP_SECN_NM': '운용사',
            'RGT_STD_DT': '권리기준일',
            'TH1_PAY_TERM_BEGIN_DT': '지급개시일',
            'ESTM_STDPRC': '주당분배금',
            'BUNBE': '분배율',
            'TAXSTD': '과세표준가',
            'RGT_RSN_DTAIL_NM': '분배사유',
            'ETF_SORT_CD': '유형코드',
            'ISSUCO_CUSTNO': '발행회사번호'
        }

        df = df.rename(columns=column_mapping)

        # 날짜 형식 변환
        date_cols = ['권리기준일', '지급개시일']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')

        # 숫자 형식 변환
        numeric_cols = ['주당분배금', '과세표준가']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        return df


def run():
    """ETF 분배금 크롤러 실행"""
    print("=" * 60)
    print("ETF 분배금 데이터 크롤러")
    print("=" * 60)

    crawler = ETFDividendCrawler()

    df = crawler.fetch_all(
        from_date=config.DIVIDEND_START_DATE,
        to_date=datetime.now().strftime('%Y%m%d'),
        max_pages=200
    )

    if df.empty:
        print("수집된 데이터가 없습니다.")
        return None

    print(f"\n총 {len(df)}건 수집 완료")
    print(f"\n컬럼: {df.columns.tolist()}")
    print(f"\n샘플 데이터:")
    print(df.head(10))

    # 저장
    output_file = os.path.join(config.OUTPUT_DIR, f"etf_dividend_{datetime.now().strftime('%Y%m%d')}.xlsx")
    df.to_excel(output_file, index=False)
    print(f"\n저장 완료: {output_file}")

    return df


if __name__ == "__main__":
    run()
