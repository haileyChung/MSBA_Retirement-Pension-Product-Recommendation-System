"""
리츠 배당 데이터 크롤러
- SEIBRO에서 리츠 배당내역 수집
- 종목명(KOR_SECN_NM)으로 검색하여 주당배당액(CASH_ALOC_AMT) 수집
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
from datetime import datetime
import time
import os

import config


class ReitsDividendCrawler:
    """리츠 배당 데이터 수집기"""

    def __init__(self):
        self.url = "https://seibro.or.kr/websquare/engine/proworks/callServletService.jsp"
        self.headers = {
            "Content-Type": "application/xml; charset=UTF-8",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Referer": "https://seibro.or.kr/websquare/control.jsp?w2xPath=/IPORTAL/user/company/BIP_CNTS01041V.xml&menuNo=285",
            "Origin": "https://seibro.or.kr"
        }
        self.page_size = 100

    def _build_request_xml(self, kor_secn_nm, start_date, end_date, start_page=1, end_page=100):
        """API 요청 XML 생성 - 종목명으로 검색"""
        return f'''<reqParam action="divStatInfoPList" task="ksd.safe.bip.cnts.Company.process.EntrFnafInfoPTask">
    <RGT_STD_DT_FROM value="{start_date}"/>
    <RGT_STD_DT_TO value="{end_date}"/>
    <ISSUCO_CUSTNO value=""/>
    <KOR_SECN_NM value="{kor_secn_nm}"/>
    <SECN_KACD value=""/>
    <RGT_RSN_DTAIL_SORT_CD value=""/>
    <LIST_TPCD value=""/>
    <START_PAGE value="{start_page}"/>
    <END_PAGE value="{end_page}"/>
    <MENU_NO value="285"/>
    <CMM_BTN_ABBR_NM value="total_search,openall,print,hwp,word,pdf,searchIcon,seach,xls,"/>
    <W2XPATH value="/IPORTAL/user/company/BIP_CNTS01041V.xml"/>
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

    def fetch_by_name(self, kor_secn_nm, start_date, end_date):
        """
        종목명으로 배당 내역 조회

        Args:
            kor_secn_nm: 종목명 (예: 맥쿼리한국인프라투융자회사)
            start_date: 조회 시작일 (YYYYMMDD)
            end_date: 조회 종료일 (YYYYMMDD)

        Returns:
            배당 내역 리스트
        """
        payload = self._build_request_xml(kor_secn_nm, start_date, end_date)

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
                record['검색종목명'] = kor_secn_nm

            return records
        except requests.RequestException as e:
            print(f"요청 에러 ({kor_secn_nm}): {e}")
            return []

    def fetch_all_from_list(self, reits_list, start_date=None, end_date=None, delay=None):
        """
        리츠 목록 전체의 배당 내역 조회

        Args:
            reits_list: (상품코드, 상품명, KOR_SECN_NM) 튜플 리스트
            start_date: 조회 시작일 (YYYYMMDD), 기본값: config 설정값
            end_date: 조회 종료일 (YYYYMMDD), 기본값: 오늘
            delay: 요청 간 대기 시간 (초)

        Returns:
            DataFrame with all REITs dividend data
        """
        end_date = end_date or datetime.now().strftime('%Y%m%d')
        start_date = start_date or config.DIVIDEND_START_DATE
        delay = delay or config.API_DELAY

        print(f"=" * 60)
        print(f"SEIBRO 리츠 배당내역 크롤링")
        print(f"조회 기간: {start_date} ~ {end_date}")
        print(f"대상 리츠 수: {len(reits_list)}개")
        print(f"=" * 60)

        all_records = []
        success_count = 0
        no_data_count = 0

        for idx, (code, name, kor_secn_nm) in enumerate(reits_list, 1):
            records = self.fetch_by_name(kor_secn_nm, start_date, end_date)

            if records:
                for record in records:
                    record['리츠 코드'] = code
                    record['리츠명'] = name

                all_records.extend(records)
                success_count += 1
                print(f"[{idx}/{len(reits_list)}] {name} ({code}): {len(records)}건")
            else:
                no_data_count += 1
                print(f"[{idx}/{len(reits_list)}] {name} ({code}): 데이터 없음")

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
            'RGT_STD_DT': '배정기준일',
            'TH1_PAY_TERM_BEGIN_DT': '현금배당지급일',
            'DELI_DT': '주식배당교부일',
            'KN_DIV_PAY_DT': '현물배당지급일',
            'SHOTN_ISIN': '종목코드',
            'KOR_SECN_NM': '종목명_SEIBRO',
            'LIST_TPNM': '시장구분',
            'RGT_RSN_DTAIL_SORT_NM': '배당구분',
            'SECN_DTAIL_KANM': '주식종류',
            'CASH_ALOC_AMT': '주당배당액',
            'DIFF_ALOC_AMT': '차등배당액',
            'CASH_ALOC_RATIO': '현금배당률',
            'STK_ALOC_RATIO': '주식배당률',
            'DIFF_ALOC_RATIO3': '차등배당률3',
            'DIFF_ALOC_RATIO2': '차등배당률2',
            'STK_FIX_RATIO': '주식배당비율',
            'DIFF_ALOC_RATIO1': '차등배당률1',
            'ESTM_STDPRC': '주당평가금액',
            'PVAL': '액면가',
            'SETACC_MM': '결산월',
            'ISSUCO_CUSTNO': '발행회사번호',
            'RGT_RACD': '권리구분코드',
            'SETACC_MMDD': '결산월일',
            'AG_ORG_TPNM': '명의개서대리인'
        }

        df = df.rename(columns=column_mapping)

        date_cols = ['배정기준일', '현금배당지급일', '주식배당교부일', '현물배당지급일']
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], format='%Y%m%d', errors='coerce')

        numeric_cols = ['주당배당액', '현금배당률', '주식배당률', '주당평가금액', '액면가']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace(',', '')
                df[col] = pd.to_numeric(df[col], errors='coerce')

        preferred_order = ['리츠 코드', '리츠명', '종목코드', '종목명_SEIBRO', '배정기준일',
                          '현금배당지급일', '배당구분', '주식종류', '주당배당액', '현금배당률',
                          '주당평가금액', '시장구분', '결산월', '명의개서대리인']

        existing_cols = [col for col in preferred_order if col in df.columns]
        other_cols = [col for col in df.columns if col not in preferred_order]
        df = df[existing_cols + other_cols]

        return df


def run():
    """리츠 배당 크롤러 실행"""
    print("=" * 60)
    print("리츠 배당 데이터 크롤러")
    print("=" * 60)

    # 통합 리츠 목록 로드
    print("리츠 목록 로딩...")
    try:
        df_input = pd.read_excel(config.REITS_LIST_FILE)
    except Exception as e:
        print(f"리츠 목록 파일 읽기 오류: {e}")
        return None

    # (리츠코드, 리츠명, KOR_SECN_NM) 튜플 리스트 생성
    reits_list = []
    for _, row in df_input.iterrows():
        code = str(row['리츠 코드']).zfill(6)
        name = row['리츠명']
        kor_secn_nm = row['KOR_SECN_NM']

        if pd.notna(kor_secn_nm) and str(kor_secn_nm).strip():
            reits_list.append((code, name, kor_secn_nm))

    print(f"유효한 리츠 수: {len(reits_list)}개")

    # 크롤링 실행
    crawler = ReitsDividendCrawler()

    df = crawler.fetch_all_from_list(
        reits_list=reits_list,
        start_date=config.DIVIDEND_START_DATE,
        end_date=datetime.now().strftime('%Y%m%d')
    )

    if df.empty:
        print("수집된 데이터가 없습니다.")
        return None

    # 결과 저장
    output_file = os.path.join(config.OUTPUT_DIR, f"reits_dividend_{datetime.now().strftime('%Y%m%d')}.xlsx")
    df.to_excel(output_file, index=False)
    print(f"\n저장 완료: {output_file}")

    # 요약 출력
    print(f"\n컬럼: {df.columns.tolist()}")
    print(f"\n배정기준일 범위: {df['배정기준일'].min()} ~ {df['배정기준일'].max()}")
    print(f"\n리츠별 건수:")
    print(df.groupby('리츠명').size().sort_values(ascending=False))

    print(f"\n샘플 데이터 (주당배당액 포함):")
    sample_cols = ['리츠명', '배정기준일', '주당배당액', '현금배당률', '배당구분']
    print(df[sample_cols].head(10))

    return df


if __name__ == "__main__":
    run()
