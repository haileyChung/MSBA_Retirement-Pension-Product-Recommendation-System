# -*- coding: utf-8 -*-
"""
리포트 생성 모듈

포트폴리오 분석 결과를 바탕으로 4개 섹션과 타임라인으로 구성된
개인맞춤 투자 리포트를 생성합니다.

AI Agent 방식으로 섹션별 동적 검색을 수행하여 최신 정보를 반영합니다.
"""

from typing import Dict, List, Optional, TYPE_CHECKING
from openai import OpenAI

from config import ReportGenerationConfig

if TYPE_CHECKING:
    from document_retrieval import InsightsLoader, NaverNewsLoader, Insight, NewsArticle


# ==================================================================================
# 유틸리티 함수
# ==================================================================================

def format_target_return(value: float) -> str:
    """목표 수익률 포맷팅 (0.075 -> '7.5', 0.07 -> '7')"""
    pct = round(value * 1000) / 10
    if pct == int(pct):
        return f"{int(pct)}"
    return f"{pct:.1f}"


def format_insights_for_prompt(insights: List) -> str:
    """인사이트를 프롬프트용 문자열로 포맷팅"""
    if not insights:
        return "리서치 인사이트가 없습니다."

    formatted = []
    for i in insights:
        sentiment_str = f"({i.sentiment})" if i.sentiment else ""
        keywords_str = f"키워드: {', '.join(i.keywords[:5])}" if i.keywords else ""
        match_info = f"[관련도: {i.relevance_score:.0f}점, {i.match_type}]"
        rationale_str = f"분석근거: {i.rationale[:100]}..." if i.rationale else ""
        formatted.append(
            f"- [{i.date}] {sentiment_str} {match_info}\n"
            f"  요약: {i.summary[:200]}\n  {keywords_str}\n  {rationale_str}"
        )
    return "\n".join(formatted)


def format_news_for_prompt(news: List) -> str:
    """뉴스를 프롬프트용 문자열로 포맷팅"""
    if not news:
        return "최신 뉴스 정보가 없습니다."

    formatted = []
    for a in news:
        formatted.append(
            f"- [{a.get_formatted_date()}] {a.get_clean_title()}\n"
            f"  내용: {a.get_clean_description()[:150]}..."
        )
    return "\n".join(formatted)


# ==================================================================================
# 리포트 생성 엔진
# ==================================================================================

class ReportGenerator:
    """
    개인맞춤 리포트 생성 엔진

    포트폴리오 정보와 시장 분석 자료를 바탕으로 4개 섹션과 타임라인을 생성합니다.
    AI Agent 방식으로 섹션별 동적 검색을 수행합니다.
    """

    def __init__(
        self,
        client: OpenAI,
        config: ReportGenerationConfig,
        insights_loader: Optional['InsightsLoader'] = None,
        news_loader: Optional['NaverNewsLoader'] = None
    ):
        """
        Args:
            client: OpenAI 클라이언트
            config: 리포트 생성 설정
            insights_loader: 인사이트 로더 (AI Agent 재검색용)
            news_loader: 뉴스 로더 (AI Agent 재검색용)
        """
        self.client = client
        self.config = config
        self.insights_loader = insights_loader
        self.news_loader = news_loader

    def set_loaders(self, insights_loader: 'InsightsLoader', news_loader: 'NaverNewsLoader'):
        """로더 설정 (지연 초기화용)"""
        self.insights_loader = insights_loader
        self.news_loader = news_loader

    async def call_gpt(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        """GPT API 호출"""
        try:
            print(f"    [DEBUG] 시스템 프롬프트 길이: {len(system_prompt)}자")
            print(f"    [DEBUG] 사용자 프롬프트 길이: {len(user_prompt)}자")
            print(f"    [DEBUG] 요청 max_completion_tokens: {max_tokens}")

            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=max_tokens,
                temperature=self.config.temperature
            )

            content = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            usage = response.usage

            print(f"    [DEBUG] 응답 finish_reason: {finish_reason}")
            print(f"    [DEBUG] 사용 토큰 - prompt: {usage.prompt_tokens}, completion: {usage.completion_tokens}")
            print(f"    [DEBUG] 응답 content 길이: {len(content) if content else 0}자")

            if content and content.strip():
                return content.strip()
            else:
                print(f"    [WARNING] 빈 응답 받음!")
                return "[응답 생성 실패] LLM이 빈 응답을 반환했습니다."

        except Exception as e:
            print(f"[GPT API 오류] {e}")
            import traceback
            traceback.print_exc()
            return "분석 내용을 생성하는 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요."

    # ==========================================================================
    # AI Agent: 동적 검색 기능
    # ==========================================================================

    async def generate_search_queries_for_section(
        self,
        section_topic: str,
        region: str,
        theme: str,
        products_info: str,
        additional_context: str = ""
    ) -> List[str]:
        """섹션별 맞춤 검색어 생성 (AI Agent 방식)"""

        prompt = f"""당신은 금융 리서치 검색 전문가입니다.
아래 리포트 섹션에 필요한 정보를 찾기 위한 **3~5개의 구체적인 검색어**를 생성하세요.

[섹션 주제]
{section_topic}

[고객 포트폴리오 정보]
- 선호 지역: {region}
- 선호 테마: {theme}
- 주요 상품: {products_info[:300]}

[추가 컨텍스트]
{additional_context}

[검색어 생성 규칙]
1. 각 검색어는 10~30자 사이
2. 구체적이고 실행 가능한 검색어 (예: "미국 반도체 2026년 전망", "S&P500 ETF 수익률 비교")
3. 지역/테마와 관련된 최신 트렌드 반영
4. 숫자, 통계, 전망 키워드 포함
5. 한 줄에 하나씩, 검색어만 출력 (설명 불필요)

검색어를 생성하세요:"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=300,
                temperature=0.8
            )

            content = response.choices[0].message.content.strip()
            # 각 줄을 검색어로 파싱
            queries = [line.strip().strip('-•*').strip()
                       for line in content.split('\n')
                       if line.strip() and len(line.strip()) > 5]

            print(f"[검색어 생성] {section_topic}: {len(queries)}개 생성")
            for q in queries[:5]:
                print(f"  - {q}")

            return queries[:5]  # 최대 5개

        except Exception as e:
            print(f"[검색어 생성 실패] {e}, 기본 검색어 사용")
            return [f"{region} {theme} 투자", f"{theme} 전망", f"{region} 시장 분석"]

    async def search_insights_with_queries(
        self,
        queries: List[str],
        region: str,
        theme: str
    ) -> str:
        """생성된 검색어로 인사이트 재검색 (AI Agent 핵심 로직)"""
        if not self.insights_loader:
            print("[WARNING] insights_loader가 설정되지 않음, 재검색 스킵")
            return ""

        all_insights = []
        seen_ids = set()

        for query in queries:
            # 기본 지역/테마 검색
            insights = self.insights_loader.search(region=region, theme=theme, max_results=5)

            # 키워드 추가 필터링
            for insight in insights:
                if insight.insight_id not in seen_ids:
                    # 검색어와 관련성 체크
                    relevance = 0
                    query_lower = query.lower()
                    summary_lower = insight.summary.lower()

                    # 검색어 단어들이 summary에 있는지 확인
                    query_words = query_lower.split()
                    for word in query_words:
                        if len(word) > 2 and word in summary_lower:
                            relevance += 1

                    if relevance > 0:
                        insight.relevance_score += relevance * 5
                        all_insights.append(insight)
                        seen_ids.add(insight.insight_id)

        # 관련도 순으로 정렬
        all_insights.sort(key=lambda x: x.relevance_score, reverse=True)
        print(f"[인사이트 재검색] {len(all_insights)}개 발견")
        return format_insights_for_prompt(all_insights[:10])

    async def search_news_with_queries(
        self,
        queries: List[str],
        region: str,
        theme: str
    ) -> str:
        """생성된 검색어로 뉴스 재검색 (AI Agent 핵심 로직)"""
        if not self.news_loader:
            print("[WARNING] news_loader가 설정되지 않음, 재검색 스킵")
            return ""

        from document_retrieval import NewsArticle

        all_articles = []
        seen_links = set()

        for query in queries[:3]:  # 뉴스는 API 제한 고려하여 3개만
            result = self.news_loader._call_api(query, display=5, sort=self.news_loader.config.naver_news_sort)

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

        print(f"[뉴스 재검색] {len(all_articles)}개 수집")
        return format_news_for_prompt(all_articles)

    # ==========================================================================
    # 섹션 생성 함수들
    # ==========================================================================

    async def generate_summary(
        self,
        region: str,
        theme: str,
        target_return: float,
        retire_year: int,
        expected_return: float,
        var95: float,
        tdf_weight: float,
        top_products: List[str]
    ) -> str:
        """포트폴리오 요약 생성 (300자 이내)"""
        products_str = ", ".join(top_products[:5])

        # 테마 표시명 변환
        theme_display = theme if theme else "분산형"
        if theme_display.startswith("지수추종_"):
            theme_display = "지수추종"

        # 지역 표시명 변환
        region_display = region if region != "지역기타" else "기타 지역"

        system_prompt = """당신은 현대차증권의 퇴직연금 전문 PB(Private Banker)입니다.
고객의 소중한 노후 자산을 책임진다는 사명감으로, 신뢰감 있고 격조 있는 어조로 설명합니다.
가벼운 표현이나 느낌표 사용을 지양하고, 전문성과 진정성이 느껴지는 문체를 사용합니다."""

        prompt = f"""당신은 현대차증권의 퇴직연금 전문 PB입니다. 고객에게 맞춤형 포트폴리오 추천 배경을 설명하는 문구를 작성해주세요.
신뢰감 있고 전문적인 어조로 작성하되, 금융 지식이 적은 고객도 이해할 수 있어야 합니다.

[고객 선택 조건]
- 은퇴 목표: {retire_year}년
- 목표 수익률: {format_target_return(target_return)}%
- 선호 지역: {region_display}
- 선호 테마: {theme_display}

[포트폴리오 분석 결과]
- 기대 수익률: {expected_return:.2f}% (목표 대비 초과 달성)
- 손실한계선(VaR 95%): -{abs(var95):.2f}% (95% 신뢰수준에서의 예상 최대 손실)
- TDF 비중: {tdf_weight:.1f}%

[핵심 메시지 3가지 - 반드시 모두 포함]
1. 목표 초과: 고객님께서 설정하신 연 {format_target_return(target_return)}% 목표 수익률을 상회하는 {expected_return:.2f}%의 기대 수익률을 확보하였습니다.
2. 맞춤 설계: {region_display} 시장의 {theme_display} 테마를 중심으로 고객님의 투자 선호를 적극 반영하였습니다.
3. 안전장치: TDF(타겟데이트펀드)를 편입하여 {retire_year}년 은퇴시점에 맞춰 자산이 점진적으로 안정화됩니다.

[작성 규칙 - 엄격히 준수]
1. 280자 내외 (3~4문장)
2. 전문적이고 신뢰감 있는 어조 사용 ("~드립니다", "~하겠습니다" 체)
3. 느낌표(!) 사용 금지 - 차분하고 격조 있게
4. 숫자는 <strong> 태그로 강조 (예: <strong>7.5%</strong>)
5. 어려운 약어나 전문용어(VaR, CVaR 등) 사용 금지
6. 개별 상품명 언급하지 말 것
7. "손실한계선(VaR 95%)"이라는 표현은 사용 가능하되, 쉽게 풀어서 설명

요약문만 작성해주세요:"""

        return await self.call_gpt(system_prompt, prompt, self.config.summary_max_tokens)

    async def generate_section1(
        self,
        region: str,
        theme: str,
        products_info: str,
        tdf_weight: float,
        retire_year: int,
        target_return: float,
        risk_weight: float,
        safe_weight: float,
        insights_info: str,
        news_info: str
    ) -> str:
        """섹션 1: 포트폴리오 구성 상품 설명 (AI Agent 방식 적용)"""

        # ===== AI Agent: 섹션별 동적 검색 =====
        print(f"\n[섹션1] AI Agent 검색 시작...")
        section_queries = await self.generate_search_queries_for_section(
            section_topic="포트폴리오 구성 배경 및 상품 소개",
            region=region,
            theme=theme,
            products_info=products_info,
            additional_context=f"TDF {tdf_weight:.1f}%, 은퇴년도 {retire_year}년"
        )

        # 생성된 검색어로 재검색
        insights_info_updated = await self.search_insights_with_queries(section_queries, region, theme)
        news_info_updated = await self.search_news_with_queries(section_queries, region, theme)

        # 업데이트된 정보가 있으면 사용, 없으면 기존 정보 사용
        final_insights = insights_info_updated if insights_info_updated and len(insights_info_updated) > 50 else insights_info
        final_news = news_info_updated if news_info_updated and len(news_info_updated) > 50 else news_info
        # =====================================

        system_prompt = f"""당신은 현대차증권의 전문 퇴직연금 자산관리 컨설턴트입니다.
고객에게 포트폴리오 구성 상품을 쉽고 전문적으로 설명해야 합니다.

[리포트 전체 흐름에서 이 섹션의 역할]
- 이 섹션은 4개 섹션으로 구성된 리포트의 **첫 번째 섹션**입니다.
- 역할: 포트폴리오 구성 배경과 상품 소개 (도입부)
- 다음 섹션(섹션2)에서 수익률과 위험을 설명하므로, 여기서는 "왜 이렇게 구성했는지"에 집중하세요.
- 수익률/손실 수치는 간략히만 언급하고, 자세한 설명은 섹션2에 맡기세요.

[형식 규칙 - 반드시 준수]
1. 반드시 3개의 문단으로 구성하세요.
2. **각 문단 사이에는 반드시 빈 줄(공백 라인)을 하나 포함하여 시각적으로 완전히 분리하세요.** (마크다운 \\n\\n 활용)
3. 문단 시작에 '1문단:', '【1문단】' 같은 말머리 기호는 절대 붙이지 마세요. 순수하게 텍스트로만 구성합니다.
4. 전체 분량은 공백 포함 {self.config.section1_min}자~{self.config.section1_max}자 사이를 엄격히 준수하세요.

[내용 규칙]
- 상품의 '개수' 언급 금지. 반드시 '비중(%)'으로만 설명할 것.

[언어 사용 규칙]
- 반드시 한국어로만 작성하세요.
- 한자(漢字), 중국어, 일본어, 기타 외국어는 절대 사용하지 마세요.
- 단, AI, TDF, ETF, S&P500 등 금융 전문 용어/약어는 예외로 허용됩니다.

[핵심 내용 강조 규칙]
- <strong> 태그 사용: 전체 텍스트의 30~35% 정도를 강조하세요.
- 강조할 대상: 수치, 핵심 인사이트, 결론 핵심 구문
- 예시: <strong>연 7.5% 수익률</strong>, <strong>안정적인 분산투자</strong>, <strong>은퇴 목표 달성</strong>

[문단 구조]
- 1문단: {region} {theme} 테마와 {retire_year}년 은퇴 목표를 연계한 구성 배경
- 2문단: 성장추구 자산({risk_weight:.1f}%)과 안전자산({safe_weight:.1f}%)의 균형과 역할 설명
- 3문단: 이 포트폴리오가 고객에게 주는 최종적인 가치와 결론

[글 흐름 규칙]
- 전체 문단이 하나의 글처럼 자연스럽게 이어지도록 작성하세요.
- 각 문단은 이전 문단의 맥락을 이어받아 통일성 있게 전개하세요."""

        user_prompt = f"""아래 정보를 바탕으로 상담 리포트의 첫 번째 섹션을 작성해주세요.
**반드시 문단 사이에 빈 줄을 넣어 3개의 덩어리로 나누어주세요.**

[포트폴리오 정보]
- 지역/테마: {region} / {theme}
- TDF 비중: {tdf_weight:.1f}% ({retire_year}년 은퇴 목표)
- 목표 수익률: 연 {format_target_return(target_return)}%
- 성장추구/안전자산 비중: {risk_weight:.1f}% / {safe_weight:.1f}%

[구성 상품 세부]
{products_info}

[AI Agent가 검색한 최신 참고 데이터]
- 리서치: {final_insights[:400]}
- 뉴스: {final_news[:400]}

지금 바로 {self.config.section1_min}~{self.config.section1_max}자 분량의 3문단 리포트를 작성하세요."""

        result = await self.call_gpt(system_prompt, user_prompt, self.config.section1_max_tokens)
        # 문단 구분을 위해 \n\n을 <br><br>로 변환
        return result.replace("\n\n", "<br><br>").replace("\n", "<br>")

    async def generate_section2(
        self,
        target_return: float,
        expected_return: float,
        var95: float,
        gain_amount: float,
        loss_var: float,
        tdf_weight: float,
        risk_weight: float,
        insights_info: str,
        news_info: str,
        region: str = "글로벌",
        theme: str = "분산형"
    ) -> str:
        """섹션 2: 기대 손실감수수준 및 예상 수익률 (AI Agent 방식 적용)"""

        # ===== AI Agent: 섹션별 동적 검색 =====
        print(f"\n[섹션2] AI Agent 검색 시작... (지역: {region}, 테마: {theme})")
        section_queries = await self.generate_search_queries_for_section(
            section_topic="기대 수익률과 손실 위험 분석",
            region=region,
            theme=theme,
            products_info="",
            additional_context=f"목표수익률 {format_target_return(target_return)}%, 기대수익률 {expected_return:.2f}%, VaR {var95:.2f}%, 리스크 관리"
        )

        # 생성된 검색어로 재검색 (고객 선택 지역/테마 반영)
        insights_info_updated = await self.search_insights_with_queries(section_queries, region, theme)
        news_info_updated = await self.search_news_with_queries(section_queries, region, theme)

        final_insights = insights_info_updated if insights_info_updated and len(insights_info_updated) > 50 else insights_info
        final_news = news_info_updated if news_info_updated and len(news_info_updated) > 50 else news_info
        # =====================================

        system_prompt = f"""당신은 현대차증권의 "Future Shield Advisor"입니다.
고객이 이해하기 쉽도록 포트폴리오의 위험과 수익을 친절하게 설명하는 것이 목표입니다.

[리포트 전체 흐름에서 이 섹션의 역할]
- 이 섹션은 4개 섹션으로 구성된 리포트의 **두 번째 섹션**입니다.
- 역할: 기대 수익률과 손실 한계선 설명 (핵심 수치 분석)
- 이전 섹션(섹션1)에서 구성 배경을 설명했으므로, 여기서는 "얼마나 벌고 잃을 수 있는지"에 집중하세요.
- 다음 섹션(섹션3)에서 시장 전망을 다루므로, 시장 상황 언급은 최소화하세요.

[언어 사용 규칙 - 반드시 준수]
- 반드시 한국어로만 작성하세요.
- 한자(漢字), 중국어, 일본어, 기타 외국어는 절대 사용하지 마세요.
- 단, AI, TDF, ETF, VaR, CVaR, S&P500 등 금융 전문 용어/약어는 예외로 허용됩니다.

[상품 개수 관련 규칙 - 매우 중요]
- 포트폴리오를 구성하는 상품의 "개수"는 절대 언급하지 마세요.
- 상품 수가 많다는 것에 대한 부정적 언급(관리 효율성 저하 등) 절대 금지
- 반드시 "비중(%)"으로만 설명하세요

[핵심 원칙]
1. 결론 먼저: "고객님이 설정하신 목표 수익률을 초과 달성합니다"로 시작
2. 쉬운 말로: 금융 용어는 반드시 괄호 안에 쉬운 설명 추가
3. 숫자의 맥락화: "100만원 투자 시 약 OO만원" 형태로 체감되게 설명
4. 신뢰 구축: "최근 시장 데이터 기준", "시장 분석에 따르면" 등 객관적 근거 명시 (단, "현대차증권 리서치팀"은 절대 언급하지 마세요)
5. 긍정적 프레이밍: 손실 한도를 지키면서 수익을 추구한다는 점 강조

[핵심 내용 강조 규칙]
- <strong> 태그 사용: 전체 텍스트의 30~35% 정도를 강조하세요.
- 강조할 대상: 수치, 핵심 인사이트, 결론 핵심 구문
- 예시: <strong>목표 초과 달성</strong>, <strong>연 15만원 수익</strong>, <strong>-15% 이내</strong>

[금융 용어 번역 가이드 - 반드시 적용]
- VaR → "손실한계선(VaR 95%)(최악의 상황에서 예상되는 최대 손실)"
- TDF → "타겟데이트펀드(은퇴 시점에 맞춰 자동으로 위험을 줄여주는 펀드)"
- 위험자산 → "주식 등 변동성이 큰 자산"
- 기대수익률 → "목표 수익률"

[톤앤매너]
- 존댓말 사용, 따뜻하고 전문적인 어조
- 불안감 조성 금지, 솔직하되 긍정적으로
- "~입니다", "~드립니다" 형태로 마무리

★★★ 반드시 {self.config.section2_min}자 이상 {self.config.section2_max}자 이하로 작성하세요. ★★★"""

        user_prompt = f"""다음 포트폴리오가 고객님 조건에 맞는지 {self.config.section2_min}자 이상 {self.config.section2_max}자 이하로 설명해주세요.

[고객님이 설정하신 조건]
- 목표 수익률: 연 {format_target_return(target_return)}%

[이 포트폴리오의 예상 성과]
- 기대 수익률: 연 {expected_return:.2f}% (100만원 투자 시 연간 약 {gain_amount:,.0f}원 수익 기대)
- 손실한계선(VaR 95%): -{abs(var95):.2f}% (100만원 투자 시 최악의 경우 약 {loss_var:,.0f}원 손실 가능)

[포트폴리오 구성 비율]
- 타겟데이트펀드(TDF): {tdf_weight:.1f}%
- 성장추구 자산: {risk_weight:.1f}%
- 안정성 확보: {100 - risk_weight:.1f}%

[AI Agent가 검색한 최신 참고 자료]
{final_insights[:500]}

[AI Agent가 검색한 최신 시장 뉴스]
{final_news[:500]}

위 정보를 바탕으로 다음 내용을 쉽고 친절하게 설명해주세요:

1. 목표 수익률 달성 여부 (첫 문장에서 결론 제시)
   - "고객님이 설정하신 목표 수익률({format_target_return(target_return)}%)을 초과 달성합니다"로 시작

2. 수익과 위험의 균형
   - 기대 수익률과 예상 손실을 100만원 기준 금액으로 설명
   - "시장 분석에 따르면", "최근 데이터 기준" 등 객관적 근거 제시 (단, "현대차증권 리서치팀"은 절대 언급 금지)

3. 현재 시장 상황과 주의점
   - 최근 뉴스/리서치 내용을 인용하여 신뢰감 부여
   - 주의할 점은 있되, 불안감 조성 없이 객관적으로 설명

{self.config.section2_min}자 이상 {self.config.section2_max}자 이하로 작성해주세요."""

        return await self.call_gpt(system_prompt, user_prompt, self.config.section2_max_tokens)

    async def generate_section3(
        self,
        region: str,
        theme: str,
        insights_info: str,
        news_info: str
    ) -> str:
        """섹션 3: 시장 전망 (AI Agent 방식 적용)"""

        # ===== AI Agent: 섹션별 동적 검색 =====
        print(f"\n[섹션3] AI Agent 검색 시작...")
        section_queries = await self.generate_search_queries_for_section(
            section_topic="시장 전망 및 투자 환경 분석",
            region=region,
            theme=theme,
            products_info="",
            additional_context="금리, 환율, 지정학적 리스크, 섹터별 전망 포함"
        )

        # 생성된 검색어로 재검색
        insights_info_updated = await self.search_insights_with_queries(section_queries, region, theme)
        news_info_updated = await self.search_news_with_queries(section_queries, region, theme)

        final_insights = insights_info_updated if insights_info_updated and len(insights_info_updated) > 50 else insights_info
        final_news = news_info_updated if news_info_updated and len(news_info_updated) > 50 else news_info
        # =====================================

        system_prompt = f"""당신은 현대차증권의 전문 퇴직연금 자산관리 컨설턴트입니다.
고객에게 시장 전망과 투자 환경을 쉽고 전문적으로 설명해야 합니다.

[리포트 전체 흐름에서 이 섹션의 역할]
- 이 섹션은 4개 섹션으로 구성된 리포트의 **세 번째 섹션**입니다.
- 역할: 시장 전망과 투자 환경 분석 (외부 환경 분석)
- 이전 섹션들(섹션1-2)에서 포트폴리오와 수익/위험을 설명했으므로, 여기서는 "시장이 어떻게 움직일지"에 집중하세요.
- 다음 섹션(섹션4)에서 종합 평가를 하므로, 최종 결론은 여기서 내리지 마세요.

[형식 규칙 - 반드시 준수]
1. 반드시 3개의 문단으로 구성하세요.
2. **각 문단 사이에는 반드시 빈 줄(공백 라인)을 하나 포함하여 시각적으로 완전히 분리하세요.** (마크다운 \\n\\n 활용)
3. 문단 시작에 '1.', '첫째,', '【1문단】' 같은 말머리 기호는 절대 붙이지 마세요. 순수하게 텍스트로만 구성합니다.
4. 전체 분량은 공백 포함 {self.config.section3_min}자~{self.config.section3_max}자 사이를 엄격히 준수하세요.

[언어 사용 규칙]
- 반드시 한국어로만 작성하세요.
- 한자(漢字), 중국어, 일본어, 기타 외국어는 절대 사용하지 마세요.
- 단, AI, TDF, ETF, S&P500, FOMC 등 금융 전문 용어/약어는 예외로 허용됩니다.

[핵심 내용 강조 규칙]
- <strong> 태그 사용: 전체 텍스트의 30~35% 정도를 강조하세요.
- 강조할 대상: 수치, 핵심 인사이트, 결론 핵심 구문
- 예시: <strong>2.73% 상승</strong>, <strong>AI 수요 급증</strong>, <strong>장기 성장 전망</strong>

[문단 구조]
- 1문단: 현재 {region} {theme} 시장 상황 (리서치/뉴스 인용)
- 2문단: 향후 전망 및 투자 시 주의할 리스크 (금리, 환율, 매크로 등)
- 3문단: 중장기 투자 기회와 분석 결론

[글 흐름 규칙]
- 전체 문단이 하나의 글처럼 자연스럽게 이어지도록 작성하세요.
- 각 문단은 이전 문단의 맥락을 이어받아 통일성 있게 전개하세요."""

        user_prompt = f"""아래 정보를 바탕으로 상담 리포트의 세 번째 섹션(시장 전망)을 작성해주세요.
**반드시 문단 사이에 빈 줄을 넣어 3개의 덩어리로 나누어주세요.**

[투자 정보]
- 지역: {region}
- 테마: {theme}

[AI Agent가 검색한 최신 분석 자료]
{final_insights[:700]}

[AI Agent가 검색한 최신 뉴스 헤드라인]
{final_news[:500]}

[금지 사항]
- "현대차증권 리서치팀", "현대차증권에 따르면" 등 현대차증권을 주어로 사용하지 마세요.
- 대신 "시장 분석에 따르면", "최근 데이터 기준" 등 객관적 표현을 사용하세요.

지금 바로 {self.config.section3_min}~{self.config.section3_max}자 분량의 3문단 리포트를 작성하세요."""

        result = await self.call_gpt(system_prompt, user_prompt, self.config.section3_max_tokens)
        # 문단 구분을 위해 \n\n을 <br><br>로 변환
        return result.replace("\n\n", "<br><br>").replace("\n", "<br>")

    async def generate_section4(
        self,
        region: str,
        theme: str,
        retire_year: int,
        expected_return: float,
        var95: float,
        risk_weight: float,
        safe_weight: float,
        tdf_weight: float,
        insights_info: str,
        news_info: str
    ) -> str:
        """섹션 4: 종합 평가 (AI Agent 방식 적용)"""

        # ===== AI Agent: 섹션별 동적 검색 =====
        print(f"\n[섹션4] AI Agent 검색 시작...")
        section_queries = await self.generate_search_queries_for_section(
            section_topic="포트폴리오 종합 평가 및 최종 추천",
            region=region,
            theme=theme,
            products_info="",
            additional_context=f"기대수익률 {expected_return:.2f}%, VaR {var95:.2f}%, 은퇴년도 {retire_year}년, 장기투자 적합성 평가"
        )

        # 생성된 검색어로 재검색
        insights_info_updated = await self.search_insights_with_queries(section_queries, region, theme)
        news_info_updated = await self.search_news_with_queries(section_queries, region, theme)

        final_insights = insights_info_updated if insights_info_updated and len(insights_info_updated) > 50 else insights_info
        final_news = news_info_updated if news_info_updated and len(news_info_updated) > 50 else news_info
        # =====================================

        system_prompt = f"""당신은 현대차증권의 수석 자산관리 컨설턴트입니다.
고객의 퇴직연금 포트폴리오를 종합적으로 평가하고 최종 추천 의견을 제시합니다.

[리포트 전체 흐름에서 이 섹션의 역할]
- 이 섹션은 4개 섹션으로 구성된 리포트의 **마지막 섹션**입니다.
- 역할: 종합 평가 및 최종 추천 의견 (결론부)
- 이전 섹션들의 내용을 종합하여 최종 결론을 제시하세요:
  * 섹션1: 포트폴리오 구성 배경
  * 섹션2: 수익률과 위험 분석
  * 섹션3: 시장 전망
- 앞선 내용을 반복하지 말고, "그래서 결론적으로 어떤가"에 집중하세요.

[언어 사용 규칙 - 반드시 준수]
- 반드시 한국어로만 작성하세요.
- 한자(漢字), 중국어, 일본어, 기타 외국어는 절대 사용하지 마세요.
- 단, AI, TDF, ETF, VaR, CVaR, S&P500 등 금융 전문 용어/약어는 예외로 허용됩니다.

[상품 개수 관련 규칙 - 매우 중요]
- 포트폴리오를 구성하는 상품의 "개수"는 절대 언급하지 마세요.
- "OO개 상품", "총 OO종목" 등 개수 표현 금지
- 상품 수가 많다는 것에 대한 부정적 언급(관리 효율성 저하 등) 절대 금지
- 반드시 "비중(%)"으로만 설명하세요: "안전자산 비중이 60%", "TDF 비중이 30%" 등

[핵심 내용 강조 규칙]
- <strong> 태그 사용: 전체 텍스트의 30~35% 정도를 강조하세요.
- 강조할 대상: 수치, 핵심 인사이트, 결론 핵심 구문
- 예시: <strong>우수한 분산투자</strong>, <strong>장기 안정성 확보</strong>, <strong>적극 추천</strong>

★★★ 반드시 {self.config.section4_min}자 이상 {self.config.section4_max}자 이하로 작성하세요. ★★★"""

        user_prompt = f"""다음 퇴직연금 포트폴리오에 대한 종합 평가를 {self.config.section4_min}자 이상 {self.config.section4_max}자 이하로 작성해주세요.

[고객 프로필]
- 선호 지역/테마: {region} / {theme}
- 목표 은퇴연도: {retire_year}년

[포트폴리오 구성]
- TDF 비중: {tdf_weight:.1f}%
- 기대 수익률: {expected_return:.2f}%
- 손실한계선(VaR 95%): -{abs(var95):.2f}%
- 성장추구 자산: {risk_weight:.1f}%
- 안전자산: {safe_weight:.1f}%

[AI Agent가 검색한 최신 분석 자료]
{final_insights[:500]}

[AI Agent가 검색한 최신 시장 뉴스]
{final_news[:500]}

[금지 사항]
- "현대차증권 리서치팀", "현대차증권에 따르면", "현대차증권은" 등 현대차증권을 주어로 사용하지 마세요.
- 대신 "시장 분석에 따르면", "최근 데이터 기준", "전문가 분석에 따르면" 등 객관적 표현을 사용하세요.

[필수 평가 항목 - 반드시 5가지 모두 포함]
1. 기대수익률: 고객 목표 수익률 대비 달성 여부 평가
2. 안정성: 손실한계선(VaR 95%) 기반 위험 수준 평가
3. 분산투자: 성장추구/안전자산 배분의 다양성 평가
4. 장기적합성: {retire_year}년 은퇴 시점까지의 투자 적합도 평가
5. 선택조건부합: 고객 선호({region}/{theme}) 반영 여부 평가

위 5가지 항목을 자연스럽게 녹여서 종합 평가를 작성하세요."""

        return await self.call_gpt(system_prompt, user_prompt, self.config.section4_max_tokens)

    async def generate_timeline(
        self,
        retire_year: int,
        theme: str,
        region: str,
        risk_weight: float,
        tdf_weight: float,
        expected_return: float = 0.0,
        products_info: str = ""
    ) -> Dict[int, str]:
        """타임라인 설명 생성 (AI 기반 동적 생성)"""
        current_year = 2026

        # 5년 단위로 연도 생성
        years = []
        for y in range(current_year, retire_year + 1, 5):
            years.append(y)
        if years[-1] != retire_year:
            years.append(retire_year)

        # 연도별 정보 생성
        years_info = []
        for i, year in enumerate(years):
            if i == 0:
                years_info.append(f"- {year}년 (현재): 투자 시작 시점")
            elif i == len(years) - 1:
                years_info.append(f"- {year}년 (은퇴): 최종 목표 시점")
            else:
                progress = (year - current_year) / (retire_year - current_year) * 100
                years_info.append(f"- {year}년 (진행률 {progress:.0f}%): 중간 단계")

        system_prompt = """당신은 현대차증권의 퇴직연금 전문 컨설턴트입니다.
고객의 은퇴까지의 투자 여정을 시각화하는 타임라인의 각 단계별 설명을 작성합니다.

[언어 사용 규칙]
- 반드시 한국어로만 작성하세요.
- 한자(漢字), 중국어, 일본어, 기타 외국어는 절대 사용하지 마세요.
- 단, AI, TDF, ETF 등 금융 전문 용어/약어는 예외로 허용됩니다.

[작성 규칙]
- 각 연도별 설명은 한 줄로 작성 (20~35자)
- 첫 부분에 전략, · 기호 뒤에 기대효과 작성
- 해당 연도의 포트폴리오 상태와 투자 전략을 구체적으로 반영
- 테마와 관련된 구체적인 내용 포함"""

        year_format_examples = "\n".join([f"{y}년: [전략] · [기대효과]" for y in years])

        user_prompt = f"""다음 고객의 은퇴 투자 여정 타임라인 설명을 작성해주세요.

[고객 프로필]
- 선호 지역: {region}
- 선호 테마: {theme}
- 목표 은퇴연도: {retire_year}년
- 투자 기간: {retire_year - current_year}년

[현재 포트폴리오 구성]
- 성장추구 자산 비중: {risk_weight:.0f}%
- TDF 비중: {tdf_weight:.0f}%
- 안전자산 비중: {100 - risk_weight:.0f}%
- 기대 수익률: {expected_return:.1f}%

[주요 구성 상품]
{products_info[:500]}

[타임라인 연도]
{chr(10).join(years_info)}

각 연도별로 다음 형식으로 작성해주세요:
{year_format_examples}

★ 각 설명은 "[전략] · [기대효과]" 형식으로, 전체 20~35자로 작성하세요.
★ 테마({theme})와 관련된 구체적인 내용을 포함하세요.
★ 시간이 지남에 따라 위험자산 비중이 줄어들고 안전자산이 늘어나는 글라이드패스를 반영하세요."""

        try:
            response = await self.call_gpt(system_prompt, user_prompt, 500)

            # 응답 파싱
            descriptions = {}
            for line in response.strip().split('\n'):
                line = line.strip()
                if not line:
                    continue
                for year in years:
                    if f"{year}년:" in line or f"{year}년 :" in line:
                        desc_part = line.split(':', 1)[-1].strip()
                        descriptions[year] = desc_part
                        break

            # 누락된 연도가 있으면 기본값 사용
            for i, year in enumerate(years):
                if year not in descriptions:
                    if i == 0:
                        descriptions[year] = f"{theme} 중심 투자 시작 · 성장자산 {risk_weight:.0f}%로 적극 운용"
                    elif i == len(years) - 1:
                        descriptions[year] = "은퇴자금 안정화 완료 · 안전자산 중심 현금흐름 확보"
                    else:
                        progress = (year - current_year) / (retire_year - current_year)
                        if progress < 0.5:
                            descriptions[year] = f"TDF 자동 리밸런싱 · {theme} 성장 수혜 지속"
                        else:
                            descriptions[year] = "안정성 강화 단계 · 변동성 완화 및 자산 보존"

            return descriptions

        except Exception as e:
            print(f"[타임라인] AI 생성 실패, 기본값 사용: {e}")
            # 기본값 반환
            descriptions = {}
            for i, year in enumerate(years):
                if i == 0:
                    descriptions[year] = f"{theme} 중심 투자 시작 · 성장자산 {risk_weight:.0f}%로 적극 운용"
                elif i == len(years) - 1:
                    descriptions[year] = "은퇴자금 안정화 완료 · 안전자산 중심 현금흐름 확보"
                else:
                    progress = (year - current_year) / (retire_year - current_year)
                    if progress < 0.5:
                        descriptions[year] = f"TDF 자동 리밸런싱 · {theme} 성장 수혜 지속"
                    else:
                        descriptions[year] = "안정성 강화 단계 · 변동성 완화 및 자산 보존"
            return descriptions
