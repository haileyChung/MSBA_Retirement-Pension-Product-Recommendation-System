# -*- coding: utf-8 -*-
"""
문서 검색 모듈

리서치 인사이트와 뉴스를 검색하여 리포트 생성에 필요한 참고 자료를 제공합니다.
지역/테마 기반 유사도 검색을 수행합니다.
"""

import json
import urllib.request
import urllib.parse
import re
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime

from config import RetrievalConfig


# ==================================================================================
# 데이터 클래스
# ==================================================================================

@dataclass
class Insight:
    """리서치 인사이트 데이터"""
    insight_id: str
    doc_id: str
    date: str
    region: List[str]
    theme: List[str]
    keywords: List[str]
    sentiment: str
    summary: str
    rationale: str = ""
    relevance_score: float = 0.0
    match_type: str = ""


@dataclass
class NewsArticle:
    """뉴스 기사 데이터"""
    title: str
    original_link: str
    naver_link: str
    description: str
    pub_date: str

    def get_clean_title(self) -> str:
        """HTML 태그 제거된 제목"""
        clean = re.sub(r'<[^>]+>', '', self.title)
        return clean.strip()

    def get_clean_description(self) -> str:
        """HTML 태그 제거된 설명"""
        clean = re.sub(r'<[^>]+>', '', self.description)
        return clean.strip()

    def get_formatted_date(self) -> str:
        """날짜 포맷팅 (YYYY-MM-DD)"""
        try:
            dt = datetime.strptime(self.pub_date, "%a, %d %b %Y %H:%M:%S %z")
            return dt.strftime("%Y-%m-%d")
        except:
            return self.pub_date[:10] if self.pub_date else ""


# ==================================================================================
# 인사이트 로더
# ==================================================================================

class InsightsLoader:
    """
    리서치 인사이트 로더

    병합된 인사이트 JSON 파일을 로드하고 지역/테마 기반 검색을 제공합니다.
    """

    def __init__(self, insights_path: Path, config: RetrievalConfig):
        """
        Args:
            insights_path: 병합된 인사이트 JSON 경로
            config: 검색 설정
        """
        self.insights_path = insights_path
        self.config = config
        self.insights: List[Dict] = []
        self.loaded = False

    def load(self) -> bool:
        """
        JSON 파일에서 인사이트 로드

        Returns:
            로드 성공 여부
        """
        try:
            if not self.insights_path.exists():
                print(f"[인사이트] 파일 없음: {self.insights_path}")
                return False

            with open(self.insights_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.insights = data.get('insight_archive', [])
            self.loaded = True
            print(f"[인사이트] {len(self.insights)}개 인사이트 로드 완료")
            return True

        except Exception as e:
            print(f"[인사이트] 로드 오류: {e}")
            return False

    def search(self, region: str, theme: str, max_results: Optional[int] = None) -> List[Insight]:
        """
        지역/테마와 관련된 인사이트 검색

        Args:
            region: 지역 (예: '한국', '미국')
            theme: 테마 (예: '반도체', 'AI테크')
            max_results: 최대 결과 개수 (None이면 config 값 사용)

        Returns:
            관련도 순으로 정렬된 인사이트 리스트
        """
        if not self.loaded:
            self.load()

        if max_results is None:
            max_results = self.config.max_insights

        matched = []

        for item in self.insights:
            # 지역/테마 매칭 점수 계산
            score = 0
            match_reasons = []

            item_regions = item.get('region', [])
            item_themes = item.get('theme', [])

            # 지역 매칭
            if region in item_regions:
                score += self.config.region_match_score
                match_reasons.append("지역일치")

            # 테마 매칭
            if theme in item_themes:
                score += self.config.theme_match_score
                match_reasons.append("테마일치")

            # 관련 키워드 매칭
            keywords = item.get('related_keywords', [])
            if any(kw for kw in keywords if region in kw or theme in kw):
                score += self.config.keyword_match_score
                match_reasons.append("키워드매칭")

            if score > 0:
                analysis = item.get('analysis', {})
                insight = Insight(
                    insight_id=item.get('insight_id', ''),
                    doc_id=item.get('doc_id', ''),
                    date=item.get('date', ''),
                    region=item_regions,
                    theme=item_themes,
                    keywords=keywords[:10],
                    sentiment=analysis.get('sentiment', ''),
                    summary=analysis.get('summary', ''),
                    rationale=analysis.get('rationale', ''),
                    relevance_score=score,
                    match_type=", ".join(match_reasons)
                )
                matched.append(insight)

        # 관련도 점수로 정렬
        matched.sort(key=lambda x: x.relevance_score, reverse=True)
        return matched[:max_results]


# ==================================================================================
# 네이버 뉴스 로더
# ==================================================================================

class NaverNewsLoader:
    """
    네이버 뉴스 검색 API 래퍼

    지역/테마에 맞는 검색 키워드로 최신 뉴스를 수집합니다.
    """

    # 지역별 검색 키워드
    REGION_KEYWORDS = {
        '미국': ['미국 증시', 'S&P500', '나스닥'],
        '중국': ['중국 증시', '상해종합', '중국 경제'],
        '한국': ['코스피', '코스닥', '한국 증시'],
        '아시아': ['아시아 증시', '일본 증시', '신흥국'],
        '지역기타': ['글로벌 증시', '유럽 증시', '신흥국'],
    }

    # 테마별 검색 키워드
    THEME_KEYWORDS = {
        'AI테크': ['AI 반도체', '인공지능 투자', '엔비디아'],
        '반도체': ['반도체 투자', '삼성전자', 'SK하이닉스'],
        '배터리전기차': ['2차전지', '전기차', '배터리 투자'],
        '바이오헬스케어': ['바이오', '헬스케어 투자', '제약'],
        '지수추종_미국': ['S&P500 ETF', '미국 지수', '나스닥100'],
        '지수추종_코스피관련': ['코스피 ETF', 'KODEX 200', '국내 지수'],
        '배당': ['배당주', '고배당 ETF', '배당 투자'],
        'ESG': ['ESG 투자', '친환경', '탄소중립'],
        '금': ['금 투자', '금 ETF', '안전자산'],
        '리츠': ['리츠 투자', 'REITs', '부동산 펀드'],
        '테마기타': ['테마 투자', '섹터 ETF', '성장주'],
        '미분류 채권': ['채권 투자', '국채', '회사채'],
        '연금 특화': ['퇴직연금', 'TDF', '연금저축'],
    }

    def __init__(self, config: RetrievalConfig):
        """
        Args:
            config: 검색 설정 (네이버 API 키 포함)
        """
        self.config = config
        self.api_url = "https://openapi.naver.com/v1/search/news.json"

    def _call_api(self, query: str, display: int = 10, sort: str = "sim") -> Dict:
        """
        네이버 API 호출

        Args:
            query: 검색 쿼리
            display: 결과 개수
            sort: 정렬 방식 (sim: 유사도순, date: 날짜순)

        Returns:
            API 응답 JSON
        """
        try:
            encoded_query = urllib.parse.quote(query)
            url = f"{self.api_url}?query={encoded_query}&display={display}&sort={sort}"

            request = urllib.request.Request(url)
            request.add_header("X-Naver-Client-Id", self.config.naver_client_id)
            request.add_header("X-Naver-Client-Secret", self.config.naver_client_secret)

            response = urllib.request.urlopen(request, timeout=10)

            if response.getcode() == 200:
                response_body = response.read().decode('utf-8')
                return json.loads(response_body)
            else:
                print(f"[네이버 뉴스 API] 오류 코드: {response.getcode()}")
                return {}

        except Exception as e:
            print(f"[네이버 뉴스 API] 오류: {e}")
            return {}

    def search(
        self,
        region: str,
        theme: str,
        max_total: Optional[int] = None
    ) -> List[NewsArticle]:
        """
        여러 쿼리로 뉴스 검색하여 통합

        Args:
            region: 지역
            theme: 테마
            max_total: 최대 결과 개수 (None이면 config 값 사용)

        Returns:
            뉴스 기사 리스트
        """
        if max_total is None:
            max_total = self.config.max_news

        all_articles = []
        seen_links = set()

        # 검색 키워드 가져오기
        region_kws = self.REGION_KEYWORDS.get(region, ['증시'])
        theme_kws = self.THEME_KEYWORDS.get(theme, ['투자'])

        # 검색 쿼리 생성
        queries = []
        queries.append(f"{region_kws[0]} {theme_kws[0]}")
        if len(theme_kws) > 1:
            queries.append(theme_kws[1])
        if len(region_kws) > 1:
            queries.append(f"{region_kws[1]} 투자")

        articles_per_query = max(3, max_total // len(queries))

        for query in queries:
            if len(all_articles) >= max_total:
                break

            result = self._call_api(
                query,
                display=articles_per_query,
                sort=self.config.naver_news_sort
            )

            for item in result.get('items', []):
                link = item.get('link', '')
                if link not in seen_links:
                    seen_links.add(link)
                    article = NewsArticle(
                        title=item.get('title', ''),
                        original_link=item.get('originallink', ''),
                        naver_link=item.get('link', ''),
                        description=item.get('description', ''),
                        pub_date=item.get('pubDate', ''),
                    )
                    all_articles.append(article)

                    if len(all_articles) >= max_total:
                        break

        print(f"[네이버 뉴스] {len(all_articles)}개 기사 수집 완료")
        return all_articles


# ==================================================================================
# 통합 검색 클래스
# ==================================================================================

class DocumentRetriever:
    """
    통합 문서 검색 클래스

    리서치 인사이트와 뉴스를 통합하여 검색합니다.
    """

    def __init__(
        self,
        insights_loader: InsightsLoader,
        news_loader: NaverNewsLoader
    ):
        """
        Args:
            insights_loader: 인사이트 로더
            news_loader: 뉴스 로더
        """
        self.insights_loader = insights_loader
        self.news_loader = news_loader

    def search(
        self,
        region: str,
        theme: str,
        include_news: bool = True
    ) -> Dict[str, List]:
        """
        지역/테마에 맞는 문서 검색

        Args:
            region: 지역
            theme: 테마
            include_news: 뉴스 포함 여부

        Returns:
            {"insights": [...], "news": [...]}
        """
        print(f"[문서 검색] 지역: {region}, 테마: {theme}")

        # 인사이트 검색
        insights = self.insights_loader.search(region, theme)
        print(f"[문서 검색] 인사이트 {len(insights)}개 발견")

        # 뉴스 검색
        news = []
        if include_news:
            news = self.news_loader.search(region, theme)
            print(f"[문서 검색] 뉴스 {len(news)}개 발견")

        return {
            "insights": insights,
            "news": news
        }
