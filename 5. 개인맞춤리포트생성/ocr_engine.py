# -*- coding: utf-8 -*-
"""
OCR 엔진 모듈

PDF 리포트를 이미지로 변환하고 GPT Vision을 사용하여 텍스트를 추출합니다.
환각 방지를 위한 엄격한 전사(transcription) 프롬프트를 사용합니다.
"""

import io
import base64
from typing import List, Optional
from pathlib import Path
from pdf2image import convert_from_path
from openai import OpenAI
from PIL import Image

from config import OCRConfig


# ==================================================================================
# OCR 엔진
# ==================================================================================

class VisionOCR:
    """
    GPT Vision을 사용한 OCR 엔진

    PDF 페이지 이미지를 텍스트로 변환합니다.
    환각 방지를 위해 '전사 기계' 페르소나를 적용하여 원문을 그대로 복사합니다.
    """

    def __init__(self, client: OpenAI, config: OCRConfig):
        """
        Args:
            client: OpenAI 클라이언트
            config: OCR 설정
        """
        self.client = client
        self.config = config

    @staticmethod
    def encode_image_to_base64(image: Image.Image) -> str:
        """
        이미지를 Base64 문자열로 인코딩

        Args:
            image: PIL Image 객체

        Returns:
            Base64 인코딩된 문자열
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _build_system_prompt(self) -> str:
        """OCR 시스템 프롬프트 생성"""
        return """
당신은 OCR(광학문자인식) 전사 기계입니다.
이미지에 보이는 텍스트를 한 글자도 바꾸지 않고 그대로 옮겨 적으세요.

[절대 금지 사항]
1. 추론하지 마세요. 글자가 불명확하면 [불명확] 표시 후 보이는 그대로 적으세요.
2. 오타를 수정하지 마세요. 원문에 오타가 있으면 그대로 유지하세요.
3. 기업명, 인명, 숫자를 절대 추측하지 마세요.
   - 예: 글자가 '마이크○'으로 보이면 '마이크론'이나 '마이크로소프트'로 추측하지 말고,
     보이는 그대로 적거나 [판독불가]로 표시하세요.
4. 당신의 금융 지식을 사용하지 마세요. 맥락상 틀려 보여도 원문 그대로 적으세요.

당신은 생각하지 않습니다. 오직 복사만 합니다.
"""

    def _build_user_prompt(self) -> str:
        """OCR 사용자 프롬프트 생성"""
        return """
이 이미지의 모든 텍스트를 그대로 전사(transcribe)하세요.

[전사 규칙]
1. 본문 텍스트: 보이는 그대로 적습니다. 한 글자도 수정/추론하지 않습니다.
2. 표(Table): 행/열 구조를 유지하며 숫자와 텍스트를 그대로 옮깁니다.
   - 형식: [표: 표제목 또는 위치 설명]
     | 열1 | 열2 | 열3 |
     | 값 | 값 | 값 |
3. 차트/그래프: 축 라벨, 범례, 수치만 읽어서 기록합니다. 해석하지 않습니다.
   - 형식: [차트: 제목] X축: ..., Y축: ..., 범례: ..., 주요수치: ...
4. 판독 불가 시: [판독불가: 대략적 위치나 문맥] 표시

절대로 내용을 해석, 요약, 수정하지 마세요. 기계처럼 복사만 하세요.
"""

    def ocr_page(self, image: Image.Image) -> str:
        """
        단일 페이지 OCR 수행

        Args:
            image: PDF 페이지 이미지

        Returns:
            추출된 텍스트
        """
        base64_image = self.encode_image_to_base64(image)

        try:
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": self._build_system_prompt()},
                    {"role": "user", "content": [
                        {"type": "text", "text": self._build_user_prompt()},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]}
                ],
                max_completion_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"    [OCR ERROR] {e}")
            return "[OCR FAILED]"

    def ocr_pdf(
        self,
        pdf_path: Path,
        poppler_path: Optional[Path] = None
    ) -> str:
        """
        전체 PDF OCR 수행

        Args:
            pdf_path: PDF 파일 경로
            poppler_path: Poppler 바이너리 경로 (선택)

        Returns:
            전체 텍스트 (페이지별로 구분)
        """
        try:
            # PDF → 이미지 변환
            if poppler_path:
                images = convert_from_path(pdf_path, poppler_path=str(poppler_path))
            else:
                images = convert_from_path(pdf_path)

        except Exception as e:
            print(f"   [ERROR] PDF 변환 실패: {e}")
            return "[PDF CONVERSION FAILED]"

        print(f"   - [OCR] {len(images)}개 페이지 변환 중...")

        full_text = ""
        for i, img in enumerate(images, 1):
            # 진행상황 출력 (덮어쓰기)
            print(f"     > 페이지 {i}/{len(images)} 처리 중...", end='\r')

            page_text = self.ocr_page(img)
            full_text += f"\n--- Page {i} ---\n{page_text}\n"

        print(f"\n     > OCR 완료.")
        return full_text


# ==================================================================================
# OCR 프로세서
# ==================================================================================

class OCRProcessor:
    """
    OCR 배치 프로세싱 클래스

    여러 PDF 파일을 일괄 처리하고 결과를 저장합니다.
    """

    def __init__(
        self,
        ocr_engine: VisionOCR,
        input_folder: Path,
        output_folder: Path,
        poppler_path: Optional[Path] = None
    ):
        """
        Args:
            ocr_engine: OCR 엔진
            input_folder: 입력 PDF 폴더
            output_folder: 출력 텍스트 폴더
            poppler_path: Poppler 경로 (선택)
        """
        self.ocr_engine = ocr_engine
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.poppler_path = poppler_path

        # 출력 폴더 생성
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def process_single_pdf(self, pdf_path: Path) -> Optional[Path]:
        """
        단일 PDF 처리

        Args:
            pdf_path: PDF 파일 경로

        Returns:
            출력 파일 경로 (실패 시 None)
        """
        file_name = pdf_path.stem
        output_path = self.output_folder / f"{file_name}_ocr.txt"

        print(f"\n>> {file_name}")

        # OCR 수행
        text = self.ocr_engine.ocr_pdf(pdf_path, self.poppler_path)

        if "[PDF CONVERSION FAILED]" in text or "[OCR FAILED]" in text:
            print(f"   [FAIL] OCR 실패")
            return None

        # 결과 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"   [DONE] 저장 완료: {output_path.name}")
        return output_path

    def process_all_pdfs(self) -> List[Path]:
        """
        폴더 내 모든 PDF 처리

        Returns:
            성공한 출력 파일 경로 리스트
        """
        pdf_files = list(self.input_folder.glob("*.pdf"))
        print(f"[INFO] {len(pdf_files)}개 PDF 파일 발견")

        output_files = []
        for pdf_path in pdf_files:
            result = self.process_single_pdf(pdf_path)
            if result:
                output_files.append(result)

        print(f"\n[완료] {len(output_files)}/{len(pdf_files)}개 파일 처리 완료")
        return output_files
