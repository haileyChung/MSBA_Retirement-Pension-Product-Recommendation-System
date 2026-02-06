# -*- coding: utf-8 -*-
"""
데이터베이스 연결 모듈

포트폴리오 최적화 결과 DB와 상품 마스터 정보를 조회합니다.
"""

import sqlite3
import json
import pandas as pd
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass


# ==================================================================================
# 데이터 클래스
# ==================================================================================

@dataclass
class PortfolioData:
    """포트폴리오 데이터"""
    portfolio_id: int
    region: str
    theme: str
    target_return: float
    retire_year: int
    expected_return: float
    var95: float
    risk_asset_weight: float
    safe_asset_weight: float
    tdf_weight: float
    total_products: int
    top10_products: List[Dict]
    region_breakdown: Dict
    theme_breakdown: Dict


# ==================================================================================
# 포트폴리오 DB 조회
# ==================================================================================

class PortfolioDatabase:
    """
    포트폴리오 최적화 결과 DB 조회 클래스

    4단계에서 생성된 portfolio_results.db를 조회합니다.
    """

    def __init__(self, db_path: Path):
        """
        Args:
            db_path: 포트폴리오 DB 경로
        """
        self.db_path = db_path

        if not self.db_path.exists():
            raise FileNotFoundError(f"포트폴리오 DB를 찾을 수 없습니다: {db_path}")

    def query_portfolio(
        self,
        region: str,
        theme: str,
        target_return: float,
        retire_year: int
    ) -> Optional[PortfolioData]:
        """
        조건에 맞는 포트폴리오 조회

        Args:
            region: 지역
            theme: 테마
            target_return: 목표 수익률
            retire_year: 은퇴연도

        Returns:
            포트폴리오 데이터 (없으면 None)
        """
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()

            cursor.execute("""
                SELECT
                    portfolio_id,
                    combo_region,
                    combo_theme,
                    combo_target_return,
                    combo_retirement_year,
                    expected_total_return_pct,
                    var_95_pct,
                    risk_asset_weight_pct,
                    tdf_weight_pct,
                    n_active_products,
                    region_breakdown_json,
                    theme_breakdown_json,
                    products_detail_json
                FROM portfolio_results
                WHERE combo_region = ?
                  AND combo_theme = ?
                  AND combo_target_return = ?
                  AND combo_retirement_year = ?
                  AND optimization_success = 1
                LIMIT 1
            """, (region, theme, target_return, retire_year))

            row = cursor.fetchone()
            conn.close()

            if not row:
                return None

            # 안전자산 비중 계산
            safe_asset_weight = 100 - (row[7] or 0)

            # 상품 목록 파싱
            products_detail = json.loads(row[12] or "[]")
            top10_products = products_detail[:10]

            return PortfolioData(
                portfolio_id=row[0],
                region=row[1],
                theme=row[2],
                target_return=row[3],
                retire_year=row[4],
                expected_return=row[5],
                var95=row[6],
                risk_asset_weight=row[7],
                safe_asset_weight=safe_asset_weight,
                tdf_weight=row[8],
                total_products=row[9],
                top10_products=top10_products,
                region_breakdown=json.loads(row[10] or "{}"),
                theme_breakdown=json.loads(row[11] or "{}")
            )

        except sqlite3.Error as e:
            print(f"[DB 오류] {e}")
            return None


# ==================================================================================
# 상품 마스터 로더
# ==================================================================================

class ProductMasterLoader:
    """
    상품 마스터 정보 로더

    2단계에서 생성된 상품명.xlsx를 로드하여 상품 정보를 제공합니다.
    """

    def __init__(self, excel_path: Path):
        """
        Args:
            excel_path: 상품명.xlsx 경로
        """
        self.excel_path = excel_path
        self.products: Dict[str, Dict] = {}
        self.loaded = False

    def load(self) -> bool:
        """
        상품 마스터 로드

        Returns:
            로드 성공 여부
        """
        try:
            if not self.excel_path.exists():
                print(f"[상품 마스터] 파일 없음: {self.excel_path}")
                return False

            df = pd.read_excel(self.excel_path)

            # 코드를 키로 하는 딕셔너리 생성
            for _, row in df.iterrows():
                code = str(row.get('상품코드', ''))
                if code:
                    self.products[code] = {
                        'name': row.get('상품명', ''),
                        'type': row.get('상품유형', ''),
                        'region': row.get('지역', ''),
                        'theme': row.get('테마', ''),
                        'asset_type': row.get('자산유형', ''),
                        'is_tdf': row.get('TDF여부', 0) != 0
                    }

            self.loaded = True
            print(f"[상품 마스터] {len(self.products)}개 상품 로드 완료")
            return True

        except Exception as e:
            print(f"[상품 마스터] 로드 오류: {e}")
            return False

    def get_product_info(self, code: str) -> Optional[Dict]:
        """
        상품 정보 조회

        Args:
            code: 상품 코드

        Returns:
            상품 정보 딕셔너리 (없으면 None)
        """
        if not self.loaded:
            self.load()

        return self.products.get(str(code))

    def enrich_products(self, products: List[Dict]) -> List[Dict]:
        """
        상품 리스트에 마스터 정보 추가

        Args:
            products: 상품 리스트 (code 필드 필요)

        Returns:
            정보가 추가된 상품 리스트
        """
        if not self.loaded:
            self.load()

        enriched = []
        for product in products:
            code = str(product.get('code', ''))
            master_info = self.products.get(code, {})

            enriched_product = product.copy()
            enriched_product.update({
                'product_region': master_info.get('region', ''),
                'product_theme': master_info.get('theme', ''),
                'product_type': master_info.get('asset_type', ''),
                'is_tdf': master_info.get('is_tdf', False)
            })
            enriched.append(enriched_product)

        return enriched
