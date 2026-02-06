# -*- coding: utf-8 -*-
"""
NER 엔진 모듈

OCR 텍스트에서 구조화된 투자 인사이트를 추출합니다.
하드코딩된 분류 기준(지역/테마/자산유형)에 맞춰 엔티티를 추출합니다.
"""

import json
import re
from typing import Dict, Any, Optional, List
from pathlib import Path
from openai import OpenAI

from config import NERConfig


# ==================================================================================
# NER 엔진
# ==================================================================================

class InsightExtractorNER:
    """
    GPT를 사용한 투자 인사이트 추출 엔진

    OCR 텍스트를 분석하여 구조화된 인사이트를 추출합니다.
    엄격한 분류 규칙을 적용하여 DB 매칭이 가능한 형태로 정규화합니다.
    """

    def __init__(self, client: OpenAI, config: NERConfig):
        """
        Args:
            client: OpenAI 클라이언트
            config: NER 설정
        """
        self.client = client
        self.config = config

    def _build_system_prompt(self, report_date: Optional[str]) -> str:
        """NER 시스템 프롬프트 생성"""
        # 허용된 분류 목록 문자열화
        region_list_str = ", ".join([f"'{r}'" for r in self.config.region_choices])
        theme_list_str = ", ".join([f"'{t}'" for t in self.config.theme_choices])
        asset_list_str = ", ".join([f"'{a}'" for a in self.config.asset_types])

        default_region = "지역기타"
        default_theme = "테마기타"

        return f"""
당신은 금융 데이터베이스 정규화 전문가입니다.
주어진 OCR 텍스트에서 투자 인사이트를 추출하되, **반드시 아래 제공된 '허용된 목록(Controlled Vocabulary)' 내에서만 분류**해야 합니다.

[★ 매우 중요: 엄격한 분류 규칙]
1. **Region (지역)**: 오직 다음 목록에 있는 단어만 사용할 수 있습니다.
   [{region_list_str}]
   - 예: 본문에 "베트남"이 있어도 목록에 없다면 '{default_region}'을 선택해야 합니다.

2. **Theme (테마)**: 오직 다음 목록에 있는 단어만 사용할 수 있습니다.
   [{theme_list_str}]
   - 예: 본문에 "HBM"이 나와도 목록에 "반도체"만 있다면 "반도체"를 선택하세요.
   - 예: "로봇"이 나와도 목록에 없다면 "AI테크"나 "{default_theme}" 등 가장 유사한 것을 선택하세요. 목록 외 단어 창조 금지.

3. **Asset Type (주식채권구분)**: 오직 다음 목록만 사용하세요.
   [{asset_list_str}]

[추출 목표]
- 고객의 펀드 포트폴리오(ETF/TDF) 수익률 변동 원인을 설명할 수 있는 '시장/섹터/거시경제' 정보를 추출하세요.
- 개별 기업(Ticker)보다는 그 기업이 속한 **테마의 방향성**을 추출하는 것이 핵심입니다.

[표/차트 해석 지침]
- OCR 텍스트에 [표: ...] 또는 [차트: ...] 형식의 데이터가 있으면, 이를 투자 관점에서 해석하세요.
- 표 데이터: 섹터별 등락률, ETF 성과 등을 분석하여 어떤 테마가 강세/약세인지 판단
- 차트 데이터: 추세(trend), 수준(level), 변동성(volatility), 의미(implication)를 분석
- 해석 결과는 해당 insight의 "analysis.rationale"에 포함시키세요.

[JSON 출력 스키마]
{{
  "doc_meta": {{
    "file_name": "문서 파일명",
    "report_date": "{report_date if report_date else 'Unknown'}",
    "page_count": 0
  }},
  "insight_archive": [
    {{
      "insight_id": "자동생성ID",
      "date": "{report_date if report_date else 'Unknown'}",
      "region": ["목록 중 선택 1개 이상"],
      "theme": ["목록 중 선택 1개 이상"],
      "asset_type": "목록 중 선택 (주식/채권/기타)",
      "content_type": "sector_view/market_strategy 등",
      "related_keywords": ["본문 핵심 키워드"],
      "analysis": {{
        "sentiment": "Positive/Negative/Neutral",
        "summary": "핵심 요약 (한글)",
        "rationale": "판단 근거 (표/차트 해석 포함)"
      }},
      "source": {{
        "page_hint": [1],
        "raw_snippet": "원문 일부..."
      }}
    }}
  ]
}}
"""

    def extract_insights(
        self,
        ocr_text: str,
        file_name: str,
        report_date: Optional[str],
        page_count: int
    ) -> Dict[str, Any]:
        """
        OCR 텍스트에서 인사이트 추출

        Args:
            ocr_text: OCR 결과 텍스트
            file_name: 파일명
            report_date: 리포트 날짜 (YYYY-MM-DD)
            page_count: 페이지 수

        Returns:
            추출된 인사이트 JSON
        """
        system_prompt = self._build_system_prompt(report_date)

        user_prompt = f"""
파일명: {file_name}
데이터 추출을 시작하세요. 위에서 정의한 [엄격한 분류 규칙]을 위반하면 DB 저장 시 오류가 발생합니다.
목록에 없는 지역/테마는 과감히 '기타' 카테고리로 분류하세요.

[추가 지시사항]
- OCR 텍스트에 포함된 [표: ...] 및 [차트: ...] 데이터를 투자 관점에서 해석하세요.
- 표의 섹터별/ETF별 등락률 데이터를 분석하여 강세/약세 테마를 파악하세요.
- 해석 결과는 관련 insight의 rationale에 포함시키세요.

--- OCR TEXT ---
{ocr_text}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            # 메타데이터 보강
            if "doc_meta" not in result:
                result["doc_meta"] = {}

            result["doc_meta"].update({
                "file_name": file_name,
                "report_date": report_date or "Unknown",
                "page_count": page_count
            })

            return result

        except Exception as e:
            print(f"    [NER ERROR] {e}")
            return {
                "doc_meta": {
                    "file_name": file_name,
                    "error": str(e)
                },
                "insight_archive": []
            }


# ==================================================================================
# NER 프로세서
# ==================================================================================

class NERProcessor:
    """
    NER 배치 프로세싱 클래스

    OCR 결과 파일들을 읽어서 인사이트를 추출하고 저장합니다.
    """

    def __init__(
        self,
        ner_engine: InsightExtractorNER,
        ocr_output_folder: Path,
        ner_output_folder: Path
    ):
        """
        Args:
            ner_engine: NER 엔진
            ocr_output_folder: OCR 출력 폴더
            ner_output_folder: NER 출력 폴더
        """
        self.ner_engine = ner_engine
        self.ocr_output_folder = ocr_output_folder
        self.ner_output_folder = ner_output_folder

        # 출력 폴더 생성
        self.ner_output_folder.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def extract_date_from_filename(file_name: str) -> Optional[str]:
        """
        파일명에서 날짜 추출 (YYYYMMDD 형식)

        Args:
            file_name: 파일명

        Returns:
            날짜 문자열 (YYYY-MM-DD) 또는 None
        """
        match = re.search(r"(\d{4})(\d{2})(\d{2})", file_name)
        if match:
            return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
        return None

    def process_single_ocr(self, ocr_file_path: Path) -> Optional[Path]:
        """
        단일 OCR 파일 처리

        Args:
            ocr_file_path: OCR 텍스트 파일 경로

        Returns:
            출력 파일 경로 (실패 시 None)
        """
        file_name = ocr_file_path.stem.replace("_ocr", "")
        output_path = self.ner_output_folder / f"{file_name}_ner.json"

        print(f"\n>> {file_name}")

        # OCR 텍스트 로드
        try:
            with open(ocr_file_path, 'r', encoding='utf-8') as f:
                ocr_text = f.read()
        except Exception as e:
            print(f"   [ERROR] 파일 읽기 실패: {e}")
            return None

        # 날짜 추출
        report_date = self.extract_date_from_filename(file_name)

        # 페이지 수 추정
        page_count = ocr_text.count("--- Page ")

        # NER 수행
        print(f"   - [NER] 인사이트 추출 중...")
        insights = self.ner_engine.extract_insights(
            ocr_text, file_name, report_date, page_count
        )

        # 결과 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(insights, f, ensure_ascii=False, indent=2)

        insight_count = len(insights.get('insight_archive', []))
        print(f"   [DONE] {insight_count}개 인사이트 추출 완료")

        return output_path

    def process_all_ocr_files(self) -> List[Path]:
        """
        폴더 내 모든 OCR 파일 처리

        Returns:
            성공한 출력 파일 경로 리스트
        """
        ocr_files = list(self.ocr_output_folder.glob("*_ocr.txt"))
        print(f"[INFO] {len(ocr_files)}개 OCR 파일 발견")

        output_files = []
        for ocr_path in ocr_files:
            result = self.process_single_ocr(ocr_path)
            if result:
                output_files.append(result)

        print(f"\n[완료] {len(output_files)}/{len(ocr_files)}개 파일 처리 완료")
        return output_files

    def merge_insights(self, output_path: Path) -> bool:
        """
        모든 NER 결과를 하나의 JSON으로 병합

        Args:
            output_path: 병합 결과 저장 경로

        Returns:
            성공 여부
        """
        ner_files = list(self.ner_output_folder.glob("*_ner.json"))
        print(f"[INFO] {len(ner_files)}개 NER 파일 병합 중...")

        all_insights = []
        for ner_file in ner_files:
            try:
                with open(ner_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    insights = data.get('insight_archive', [])

                    # doc_id 추가
                    doc_id = ner_file.stem.replace("_ner", "")
                    for insight in insights:
                        insight['doc_id'] = doc_id
                        if 'insight_id' not in insight:
                            insight['insight_id'] = f"{doc_id}_{len(all_insights)}"

                    all_insights.extend(insights)

            except Exception as e:
                print(f"   [WARNING] {ner_file.name} 병합 실패: {e}")

        # 병합 결과 저장
        merged_data = {
            "total_insights": len(all_insights),
            "source_files": len(ner_files),
            "insight_archive": all_insights
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)

        print(f"[완료] {len(all_insights)}개 인사이트 병합 완료: {output_path}")
        return True
