# -*- coding: utf-8 -*-
"""
오케스트레이터 모듈

Multi-AI Agent 시스템의 중앙 조율자입니다.
3개의 전문 에이전트(DB 적재, 문서 검색, 리포트 생성)를 조율하여
개인맞춤 투자 리포트를 생성합니다.
"""

import asyncio
import json
from typing import Dict, Optional
from pathlib import Path
from openai import OpenAI

from config import Config
from ocr_engine import VisionOCR, OCRProcessor
from ner_engine import InsightExtractorNER, NERProcessor
from document_retrieval import InsightsLoader, NaverNewsLoader, DocumentRetriever
from report_generator import ReportGenerator, format_insights_for_prompt, format_news_for_prompt
from database import PortfolioDatabase, ProductMasterLoader, PortfolioData


# ==================================================================================
# 오케스트레이터
# ==================================================================================

class ReportOrchestrator:
    """
    리포트 생성 워크플로우 오케스트레이터

    3개의 Agent를 조율하여 개인맞춤 리포트를 생성합니다:
    1. DB 적재 Agent: OCR + NER로 리서치 인사이트 추출
    2. 문서 검색 Agent: 유사도 기반 문서 검색
    3. 리포트 생성 Agent: 4개 섹션 + 타임라인 생성 (AI Agent 동적 재검색 포함)
    """

    def __init__(self, config: Config):
        """
        Args:
            config: 전체 설정
        """
        self.config = config
        self.client = OpenAI(api_key=config.api.openai_api_key)

        # Agent 초기화
        self._init_agents()

        # DB 로더 초기화
        self._init_databases()

    def _init_agents(self):
        """Agent 초기화"""
        print("[오케스트레이터] Agent 초기화 중...")

        # Agent 1: OCR + NER (DB 적재)
        self.ocr_engine = VisionOCR(self.client, self.config.ocr)
        self.ner_engine = InsightExtractorNER(self.client, self.config.ner)

        # Agent 2: 문서 검색
        self.insights_loader = InsightsLoader(
            self.config.path.output_insights_db,
            self.config.retrieval
        )
        self.news_loader = NaverNewsLoader(self.config.retrieval)
        self.document_retriever = DocumentRetriever(
            self.insights_loader,
            self.news_loader
        )

        # Agent 3: 리포트 생성 (AI Agent 동적 재검색을 위해 loaders 전달)
        self.report_generator = ReportGenerator(
            client=self.client,
            config=self.config.report,
            insights_loader=self.insights_loader,  # ← AI Agent 재검색용
            news_loader=self.news_loader           # ← AI Agent 재검색용
        )

        print("[오케스트레이터] Agent 초기화 완료")

    def _init_databases(self):
        """DB 로더 초기화"""
        print("[오케스트레이터] DB 로더 초기화 중...")

        # 포트폴리오 DB
        self.portfolio_db = PortfolioDatabase(self.config.path.portfolio_db_path)

        # 상품 마스터
        self.product_master = ProductMasterLoader(self.config.path.product_master_path)
        self.product_master.load()

        print("[오케스트레이터] DB 로더 초기화 완료")

    # ==============================================================================
    # Phase 1: DB 적재 (OCR + NER Agent)
    # ==============================================================================

    def run_ocr_ner_pipeline(self) -> bool:
        """
        Phase 1: OCR + NER 파이프라인 실행

        PDF 리서치 리포트를 읽어서 인사이트를 추출하고 DB에 저장합니다.

        Returns:
            성공 여부
        """
        print("\n" + "=" * 80)
        print("[Phase 1] DB 적재 Agent (OCR + NER)")
        print("=" * 80)

        # OCR 프로세서
        ocr_processor = OCRProcessor(
            self.ocr_engine,
            self.config.path.input_pdf_folder,
            self.config.path.output_ocr_folder,
            self.config.path.poppler_path
        )

        # NER 프로세서
        ner_processor = NERProcessor(
            self.ner_engine,
            self.config.path.output_ocr_folder,
            self.config.path.output_ner_folder
        )

        # 1. OCR 수행
        ocr_results = ocr_processor.process_all_pdfs()
        if not ocr_results:
            print("[Phase 1] OCR 실패: PDF 파일 없음 또는 오류")
            return False

        # 2. NER 수행
        ner_results = ner_processor.process_all_ocr_files()
        if not ner_results:
            print("[Phase 1] NER 실패: OCR 파일 없음 또는 오류")
            return False

        # 3. 인사이트 병합
        merge_success = ner_processor.merge_insights(self.config.path.output_insights_db)
        if not merge_success:
            print("[Phase 1] 인사이트 병합 실패")
            return False

        # 4. 인사이트 로더 리로드
        self.insights_loader.load()

        print("[Phase 1] 완료\n")
        return True

    # ==============================================================================
    # Phase 2: 문서 검색 Agent
    # ==============================================================================

    def search_documents(
        self,
        region: str,
        theme: str,
        include_news: bool = True
    ) -> Dict:
        """
        Phase 2: 문서 검색 Agent 실행

        지역/테마에 맞는 리서치 인사이트와 뉴스를 검색합니다.

        Args:
            region: 선호 지역
            theme: 선호 테마
            include_news: 뉴스 포함 여부

        Returns:
            {"insights": [...], "news": [...]}
        """
        print("\n" + "=" * 80)
        print("[Phase 2] 문서 검색 Agent")
        print("=" * 80)

        results = self.document_retriever.search(region, theme, include_news)

        print(f"[Phase 2] 완료: 인사이트 {len(results['insights'])}개, 뉴스 {len(results['news'])}개\n")
        return results

    # ==============================================================================
    # Phase 3: 리포트 생성 Agent (AI Agent 동적 재검색 포함)
    # ==============================================================================

    async def generate_report(
        self,
        portfolio_data: PortfolioData,
        insights: list,
        news: list
    ) -> Dict:
        """
        Phase 3: 리포트 생성 Agent 실행

        포트폴리오 데이터와 참고 자료를 바탕으로 맞춤형 리포트를 생성합니다.
        각 섹션 생성 시 AI Agent 방식으로 동적 재검색을 수행합니다.

        Args:
            portfolio_data: 포트폴리오 데이터
            insights: 리서치 인사이트
            news: 뉴스 기사

        Returns:
            {
                "summary": "...",
                "section1": "...",
                "section2": "...",
                "section3": "...",
                "section4": "...",
                "timeline": {...}
            }
        """
        print("\n" + "=" * 80)
        print("[Phase 3] 리포트 생성 Agent (AI Agent 동적 재검색 활성화)")
        print("=" * 80)

        # 초기 인사이트/뉴스 포맷팅 (섹션별 AI Agent가 동적으로 재검색함)
        insights_text = format_insights_for_prompt(insights)
        news_text = format_news_for_prompt(news)

        # 상품 정보 포맷팅
        products_info = "\n".join([
            f"- {p.get('name', '')}: {p.get('weight_pct', 0):.1f}%"
            for p in portfolio_data.top10_products
        ])

        # 요약 생성
        print("[Phase 3] 요약 생성 중...")
        summary = await self.report_generator.generate_summary(
            portfolio_data.region,
            portfolio_data.theme,
            portfolio_data.target_return,
            portfolio_data.retire_year,
            portfolio_data.expected_return,
            portfolio_data.var95,
            portfolio_data.tdf_weight,
            [p['name'] for p in portfolio_data.top10_products[:5]]
        )

        # 섹션 1 생성 (AI Agent 동적 재검색 수행)
        print("[Phase 3] 섹션 1 생성 중... (AI Agent 동적 재검색)")
        section1 = await self.report_generator.generate_section1(
            portfolio_data.region,
            portfolio_data.theme,
            products_info,
            portfolio_data.tdf_weight,
            portfolio_data.retire_year,
            portfolio_data.target_return,
            portfolio_data.risk_asset_weight,
            portfolio_data.safe_asset_weight,
            insights_text,
            news_text
        )

        # 섹션 2 생성 (AI Agent 동적 재검색 수행)
        print("[Phase 3] 섹션 2 생성 중... (AI Agent 동적 재검색)")
        gain_amount = portfolio_data.expected_return * 10000
        loss_amount = abs(portfolio_data.var95) * 10000

        section2 = await self.report_generator.generate_section2(
            portfolio_data.target_return,
            portfolio_data.expected_return,
            portfolio_data.var95,
            gain_amount,
            loss_amount,
            portfolio_data.tdf_weight,
            portfolio_data.risk_asset_weight,
            insights_text,
            news_text,
            region=portfolio_data.region,  # AI Agent 재검색에 필요
            theme=portfolio_data.theme     # AI Agent 재검색에 필요
        )

        # 섹션 3 생성 (AI Agent 동적 재검색 수행)
        print("[Phase 3] 섹션 3 생성 중... (AI Agent 동적 재검색)")
        section3 = await self.report_generator.generate_section3(
            portfolio_data.region,
            portfolio_data.theme,
            insights_text,
            news_text
        )

        # 섹션 4 생성 (AI Agent 동적 재검색 수행)
        print("[Phase 3] 섹션 4 생성 중... (AI Agent 동적 재검색)")
        section4 = await self.report_generator.generate_section4(
            portfolio_data.region,
            portfolio_data.theme,
            portfolio_data.retire_year,
            portfolio_data.expected_return,
            portfolio_data.var95,
            portfolio_data.risk_asset_weight,
            portfolio_data.safe_asset_weight,
            portfolio_data.tdf_weight,
            insights_text,
            news_text
        )

        # 타임라인 생성
        print("[Phase 3] 타임라인 생성 중...")
        timeline = await self.report_generator.generate_timeline(
            portfolio_data.retire_year,
            portfolio_data.theme,
            portfolio_data.region,
            portfolio_data.risk_asset_weight,
            portfolio_data.tdf_weight,
            expected_return=portfolio_data.expected_return,
            products_info=products_info
        )

        print("[Phase 3] 완료\n")

        return {
            "summary": summary,
            "section1": section1,
            "section2": section2,
            "section3": section3,
            "section4": section4,
            "timeline": timeline
        }

    # ==============================================================================
    # 통합 워크플로우
    # ==============================================================================

    async def generate_personalized_report(
        self,
        region: str,
        theme: str,
        target_return: float,
        retire_year: int,
        save_to_file: bool = True
    ) -> Optional[Dict]:
        """
        개인맞춤 리포트 생성 (전체 워크플로우)

        Args:
            region: 선호 지역
            theme: 선호 테마
            target_return: 목표 수익률
            retire_year: 은퇴연도
            save_to_file: 파일 저장 여부

        Returns:
            리포트 데이터 (실패 시 None)
        """
        print("\n" + "=" * 80)
        print("[오케스트레이터] 개인맞춤 리포트 생성 시작")
        print("=" * 80)
        print(f"  - 지역: {region}")
        print(f"  - 테마: {theme}")
        print(f"  - 목표 수익률: {target_return * 100:.1f}%")
        print(f"  - 은퇴연도: {retire_year}년")
        print()

        # 1. 포트폴리오 데이터 조회
        print("[Step 1] 포트폴리오 데이터 조회 중...")
        portfolio_data = self.portfolio_db.query_portfolio(
            region, theme, target_return, retire_year
        )

        if not portfolio_data:
            print("[오류] 해당 조건의 포트폴리오를 찾을 수 없습니다.")
            return None

        print(f"[Step 1] 포트폴리오 발견: ID={portfolio_data.portfolio_id}\n")

        # 2. 문서 검색 (Phase 2) - 초기 검색
        doc_results = self.search_documents(region, theme, include_news=True)

        # 3. 리포트 생성 (Phase 3) - AI Agent가 섹션별로 동적 재검색 수행
        report = await self.generate_report(
            portfolio_data,
            doc_results['insights'],
            doc_results['news']
        )

        # 4. 결과 저장
        if save_to_file:
            output_file = (
                self.config.path.output_reports_folder /
                f"report_{region}_{theme}_{retire_year}.json"
            )
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

            print(f"[완료] 리포트 저장: {output_file}")

        print("\n" + "=" * 80)
        print("[오케스트레이터] 리포트 생성 완료")
        print("=" * 80 + "\n")

        return report
