# -*- coding: utf-8 -*-
"""
유틸리티 함수 모듈

공통 유틸리티 함수를 제공합니다.
"""

import re
from pathlib import Path
from typing import Optional


# ==================================================================================
# 날짜 관련 유틸리티
# ==================================================================================

def extract_date_from_filename(filename: str) -> Optional[str]:
    """
    파일명에서 날짜 추출 (YYYYMMDD → YYYY-MM-DD)

    Args:
        filename: 파일명

    Returns:
        날짜 문자열 (YYYY-MM-DD) 또는 None

    Examples:
        >>> extract_date_from_filename("report_20250204_final.pdf")
        '2025-02-04'
    """
    match = re.search(r"(\d{4})(\d{2})(\d{2})", filename)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return None


# ==================================================================================
# 텍스트 정리 유틸리티
# ==================================================================================

def clean_html_tags(text: str) -> str:
    """
    HTML 태그 제거

    Args:
        text: HTML이 포함된 텍스트

    Returns:
        태그가 제거된 텍스트

    Examples:
        >>> clean_html_tags("<strong>test</strong>")
        'test'
    """
    clean = re.sub(r'<[^>]+>', '', text)
    return clean.strip()


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    텍스트 길이 제한

    Args:
        text: 원본 텍스트
        max_length: 최대 길이
        suffix: 말줄임표 (기본값: "...")

    Returns:
        잘린 텍스트

    Examples:
        >>> truncate_text("Hello World", 5)
        'Hello...'
    """
    if len(text) <= max_length:
        return text

    return text[:max_length] + suffix


# ==================================================================================
# 파일 관련 유틸리티
# ==================================================================================

def ensure_dir(path: Path) -> Path:
    """
    디렉토리가 없으면 생성

    Args:
        path: 디렉토리 경로

    Returns:
        생성된 (또는 기존) 경로
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_output_filename(
    input_path: Path,
    suffix: str,
    output_dir: Path,
    extension: str = ".txt"
) -> Path:
    """
    출력 파일명 생성

    Args:
        input_path: 입력 파일 경로
        suffix: 접미사 (예: "_ocr", "_ner")
        output_dir: 출력 디렉토리
        extension: 확장자 (기본값: ".txt")

    Returns:
        출력 파일 경로

    Examples:
        >>> get_output_filename(Path("report.pdf"), "_ocr", Path("output"))
        Path('output/report_ocr.txt')
    """
    stem = input_path.stem
    return output_dir / f"{stem}{suffix}{extension}"


# ==================================================================================
# 포맷팅 유틸리티
# ==================================================================================

def format_percentage(value: float, decimal_places: int = 2) -> str:
    """
    비율을 백분율 문자열로 변환

    Args:
        value: 비율 (0.075 → 7.5%)
        decimal_places: 소수점 자릿수

    Returns:
        백분율 문자열

    Examples:
        >>> format_percentage(0.075)
        '7.50%'
        >>> format_percentage(0.07, 1)
        '7.0%'
    """
    pct = value * 100
    return f"{pct:.{decimal_places}f}%"


def format_currency(amount: float, currency: str = "원") -> str:
    """
    금액 포맷팅

    Args:
        amount: 금액
        currency: 통화 단위

    Returns:
        포맷된 금액 문자열

    Examples:
        >>> format_currency(1000000)
        '1,000,000원'
    """
    return f"{amount:,.0f}{currency}"


# ==================================================================================
# 진행 표시 유틸리티
# ==================================================================================

class ProgressTracker:
    """
    간단한 진행 상황 추적 클래스

    콘솔에 진행 상황을 표시합니다.
    """

    def __init__(self, total: int, description: str = "처리 중"):
        """
        Args:
            total: 전체 작업 수
            description: 작업 설명
        """
        self.total = total
        self.current = 0
        self.description = description

    def update(self, increment: int = 1):
        """진행 상황 업데이트"""
        self.current += increment
        self._print_progress()

    def _print_progress(self):
        """진행 상황 출력"""
        percent = (self.current / self.total) * 100 if self.total > 0 else 0
        print(f"\r{self.description}: {self.current}/{self.total} ({percent:.1f}%)", end='')

        if self.current >= self.total:
            print()  # 완료 시 줄바꿈


# ==================================================================================
# 검증 유틸리티
# ==================================================================================

def validate_region(region: str, allowed_regions: list) -> bool:
    """
    지역 값 검증

    Args:
        region: 지역
        allowed_regions: 허용된 지역 리스트

    Returns:
        유효 여부
    """
    return region in allowed_regions


def validate_theme(theme: str, allowed_themes: list) -> bool:
    """
    테마 값 검증

    Args:
        theme: 테마
        allowed_themes: 허용된 테마 리스트

    Returns:
        유효 여부
    """
    return theme in allowed_themes


def validate_return(target_return: float, min_return: float = 0.0, max_return: float = 0.2) -> bool:
    """
    목표 수익률 검증

    Args:
        target_return: 목표 수익률
        min_return: 최소값 (기본값: 0%)
        max_return: 최대값 (기본값: 20%)

    Returns:
        유효 여부
    """
    return min_return <= target_return <= max_return
