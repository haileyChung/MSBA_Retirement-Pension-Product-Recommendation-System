# -*- coding: utf-8 -*-
"""
상품명 기반 국가/테마 분류 (LLM 활용)
=====================================
펀드 상품명을 분석하여 국가, 자산유형, 테마를 자동 분류합니다.

사용법:
    1. config.py에서 LLMConfig 설정 확인
    2. 환경변수 OPENAI_API_KEY 설정 또는 config에 직접 입력
    3. python 3_상품분류_LLM.py 실행
"""

import os
import re
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from config import LLMConfig


# =============================================================================
# 캐시 관리 클래스
# =============================================================================

class LabelCache:
    """라벨 캐시 관리 클래스"""

    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path)
        self._cache: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        """캐시 파일 로드"""
        if not self.cache_path.exists():
            return

        with self.cache_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    name = obj.get("name")
                    if name:
                        self._cache[name] = obj
                except (json.JSONDecodeError, KeyError):
                    continue

    def get(self, name: str) -> Optional[Dict[str, str]]:
        """캐시에서 라벨 조회"""
        return self._cache.get(name)

    def contains(self, name: str) -> bool:
        """캐시에 존재 여부 확인"""
        return name in self._cache

    def add(self, record: Dict[str, str]) -> None:
        """캐시에 단일 레코드 추가"""
        name = record.get("name")
        if name:
            self._cache[name] = record

    def save_batch(self, records: List[Dict[str, Any]]) -> None:
        """배치 레코드를 캐시 파일에 추가 저장"""
        if not records:
            return

        with open(self.cache_path, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                self.add(r)

    def __len__(self) -> int:
        return len(self._cache)


# =============================================================================
# LLM 클라이언트 인터페이스
# =============================================================================

class BaseLLMClient(ABC):
    """LLM 클라이언트 추상 클래스"""

    @abstractmethod
    def classify(self, names: List[str]) -> List[Dict[str, str]]:
        """상품명 리스트를 분류하여 결과 반환"""
        pass


class OpenAIClient(BaseLLMClient):
    """OpenAI API 클라이언트"""

    SYSTEM_PROMPT = """당신은 금융상품명을 보고 그 상품이 주로 노출되는 '국가/지역', '주식/채권/기타', 와'테마/섹터'를 추론하는 어시스턴트입니다.

규칙:
1) 상품명 텍스트에 근거하여 가장 타당한 국가/지역(country), 주식/채권/기타 구분(asset_type), 테마(theme)를 제안하세요.
2) 확신이 낮으면 '글로벌' 또는 '기타'처럼 광범위한 표현을 사용해도 됩니다.
3) 출력은 JSON 배열로만 반환합니다. 각 항목은 {"name": "...", "country": "...", "asset_type": "...", "theme": "..."} 형식입니다.

분류 기준 (아래 예시를 참고하되, 제한되지 않습니다):
- country: 한국, 미국, 일본, 중국, 유럽, 아시아, 지역기타 등
- asset_type: 주식, 채권, 기타 중 하나
- theme: AI테크, 반도체, 배터리전기차, ESG, 지수추종_미국, 지수추종_지역특화, 지수추종_코스피관련,
         연금_특화, 미분류_채권, 테마기타, 금, 반도체, 바이오헬스케어, 배당, 리츠, 소비재 등

참고 예시:
- "TIGER 미국테크TOP10 INDXN" → country: "미국", asset_type: "주식", theme: "AI테크"
- "KODEX 미국나스닥100" → country: "미국", asset_type: "주식", theme: "지수추종_미국"
- "ACE ESG액티브" → country: "한국", asset_type: "주식", theme: "ESG"
- "TIGER 미국나스닥100채권혼합Fn" → country: "미국", asset_type: "채권", theme: "미분류_채권"
- "ACE KRX금전융" → country: "지역기타", asset_type: "기타", theme: "금"
- "TIGER 차이나반도체FACTSET" → country: "중국", asset_type: "주식", theme: "반도체"
- "KIWOOM TDF2030액티브" → country: "지역기타", asset_type: "채권", theme: "연금_특화"
"""

    def __init__(self, api_key: str, model: str, temperature: float):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai 패키지를 설치하세요: pip install openai>=1.0.0")

        if not api_key:
            raise ValueError("OpenAI API 키가 설정되지 않았습니다.")

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature

    def _build_user_prompt(self, names: List[str]) -> str:
        """사용자 프롬프트 생성"""
        obj = {
            "instruction": "다음 상품명 리스트에 대해 국가/지역(country), 주식/채권/기타 구분(asset_type), 테마(theme)를 추론해 JSON 배열로만 반환하세요.",
            "items": names
        }
        return json.dumps(obj, ensure_ascii=False, indent=2)

    def classify(self, names: List[str]) -> List[Dict[str, str]]:
        """상품명 분류"""
        user_prompt = self._build_user_prompt(names)

        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"}
        )

        content = resp.choices[0].message.content
        results = self._parse_response(content)

        # 방어적 매칭: 입력 순서대로 결과 반환
        return self._match_results(names, results)

    def _parse_response(self, content: str) -> List[Dict[str, str]]:
        """응답 파싱"""
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list):
                        return v
        except (json.JSONDecodeError, TypeError):
            pass
        return []

    def _match_results(self, names: List[str], results: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """입력 이름과 결과 매칭"""
        out = []
        for name in names:
            rec = next(
                (r for r in results if str(r.get("name", "")).strip() == str(name).strip()),
                None
            )
            if rec is None:
                out.append({
                    "name": name,
                    "country": "기타",
                    "asset_type": "기타",
                    "theme": "기타"
                })
            else:
                out.append({
                    "name": name,
                    "country": str(rec.get("country", "기타")).strip(),
                    "asset_type": str(rec.get("asset_type", "기타")).strip(),
                    "theme": str(rec.get("theme", "기타")).strip()
                })
        return out


# =============================================================================
# 라벨 정규화 클래스
# =============================================================================

class LabelNormalizer:
    """라벨 정규화 클래스"""

    COUNTRY_PATTERNS = [
        (r"\b(korea|korean|south korea|kor|kospi|kosdaq|k-)\b", "한국"),
        (r"\b(usa|us|u\.s\.|s&p|sp500|dow|nasdaq)\b", "미국"),
        (r"\b(japan|jp|nikkei|topix)\b", "일본"),
        (r"\b(china|cn|shanghai|shenzhen|csi|h-shares)\b", "중국"),
        (r"\b(europe|eu|eurozone|stoxx)\b", "유럽"),
        (r"\b(emerging|em|latam|asean|sea|apac|asia ex[- ]?japan)\b", "신흥국"),
        (r"\b(global|world|acwi|msci all country|all country)\b", "글로벌"),
    ]

    THEME_PATTERNS = [
        (r"\b(ai|artificial intelligence|genai|테크|tech)\b", "AI테크"),
        (r"\b(robot|robotics|로봇)\b", "AI테크"),
        (r"\b(semi(conductor)?s?|chip|foundry|memory|반도체)\b", "반도체"),
        (r"\b(ev|battery|전기차|배터리|electric vehicle)\b", "배터리전기차"),
        (r"\b(esg|친환경|탄소|저탄소|renewable|clean|green|풍력|태양광)\b", "ESG"),
        (r"\b(s&p|sp500|nasdaq|나스닥|kospi|코스피|msci)\b", "지수추종_미국"),
        (r"\b(euro|stoxx|유로)\b", "지수추종_지역특화"),
        (r"\b(tdf|연금|retirement|pension)\b", "연금_특화"),
        (r"\b(bond|채권|credit|크레딧)\b", "미분류_채권"),
        (r"\b(gold|xau|precious metal|bullion|금)\b", "금"),
        (r"\b(material|commodity|원자재|metals|mining)\b", "원자재"),
        (r"\b(reit|reits|real\s?estate|property|리츠|부동산)\b", "리츠"),
        (r"\b(defense|aero(space)?|군수|방산)\b", "방산"),
        (r"\b(ship(building)?|조선|marine|naval)\b", "조선"),
        (r"\b(energy|oil|gas|석유|천연가스|uranium|에너지)\b", "에너지"),
        (r"\b(bio|healthcare|pharma|med(tech)?|바이오|헬스케어)\b", "바이오헬스케어"),
        (r"\b(fin(tech)?|bank|insurance|증권|finance|금융)\b", "금융"),
    ]

    COUNTRY_HINTS = {
        "미국": "미국", "한국": "한국", "코리아": "한국",
        "일본": "일본", "중국": "중국", "차이나": "중국",
        "유럽": "유럽", "유로": "유럽", "신흥국": "신흥국",
        "글로벌": "글로벌", "세계": "글로벌"
    }

    THEME_HINTS = [
        ("ai테크", "AI테크"), ("ai", "AI테크"), ("인공지능", "AI테크"),
        ("반도체", "반도체"),
        ("배터리전기차", "배터리전기차"), ("배터리", "배터리전기차"), ("전기차", "배터리전기차"),
        ("esg", "ESG"), ("저탄소", "ESG"),
        ("지수추종", "지수추종_미국"), ("미국", "지수추종_미국"),
        ("연금", "연금_특화"), ("tdf", "연금_특화"),
        ("채권", "미분류_채권"), ("테마기타", "테마기타"),
        ("리츠", "리츠"), ("부동산", "리츠"), ("금", "금"),
        ("방산", "방산"), ("조선", "조선"),
        ("바이오", "바이오헬스케어"), ("헬스케어", "바이오헬스케어"),
        ("에너지", "에너지"), ("원자재", "원자재"), ("금융", "금융"),
    ]

    @staticmethod
    def _normalize_text(s: str) -> str:
        """텍스트 정규화"""
        s = s.lower()
        s = re.sub(r"[\[\]\(\)\-_/]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def normalize_country(self, label: str) -> str:
        """국가 라벨 정규화"""
        if not label or not str(label).strip():
            return "기타"

        lab = self._normalize_text(str(label))

        for pat, repl in self.COUNTRY_PATTERNS:
            if re.search(pat, lab, flags=re.IGNORECASE):
                return repl

        for hint, value in self.COUNTRY_HINTS.items():
            if hint in label:
                return value

        return label.strip()

    def normalize_theme(self, label: str) -> str:
        """테마 라벨 정규화"""
        if not label or not str(label).strip():
            return "기타"

        lab = self._normalize_text(str(label))

        for pat, repl in self.THEME_PATTERNS:
            if re.search(pat, lab, flags=re.IGNORECASE):
                return repl

        for hint, value in self.THEME_HINTS:
            if hint in lab:
                return value

        return label.strip()

    def normalize_asset_type(self, label: str) -> str:
        """자산유형 라벨 정규화"""
        if not label or not str(label).strip():
            return "기타"

        lab = str(label).strip().lower()

        if "주식" in lab or "equity" in lab or "stock" in lab:
            return "주식"
        if "채권" in lab or "bond" in lab or "credit" in lab:
            return "채권"

        return "기타"


# =============================================================================
# 빈도 기반 정제 클래스
# =============================================================================

class FrequencyRefiner:
    """빈도 기반 라벨 정제 클래스"""

    def __init__(self, topn: int, min_group_size: int):
        self.topn = topn
        self.min_group_size = min_group_size

    def refine(self, series: pd.Series) -> pd.Series:
        """상위 N개 유지, 나머지는 '기타'로 변환"""
        vc = series.value_counts()
        top_labels = set(vc.head(self.topn).index.tolist())

        # 상위 N개만 유지
        reduced = series.apply(lambda x: x if x in top_labels else "기타")

        # 최소 그룹 크기 미만은 '기타'로
        vc2 = reduced.value_counts()
        small = set(vc2[vc2 < self.min_group_size].index.tolist()) - {"기타"}
        reduced = reduced.apply(lambda x: "기타" if x in small else x)

        return reduced


# =============================================================================
# 메인 분류 파이프라인
# =============================================================================

class ProductClassifier:
    """상품 분류 파이프라인"""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.cache = LabelCache(config.cache_file)
        self.normalizer = LabelNormalizer()
        self.country_refiner = FrequencyRefiner(config.topn_country, config.min_group_size)
        self.theme_refiner = FrequencyRefiner(config.topn_theme, config.min_group_size)

        # API 키 설정 (환경변수 우선)
        api_key = os.getenv("OPENAI_API_KEY") or config.openai_api_key
        self.llm_client = OpenAIClient(api_key, config.model_name, config.temperature)

    def load_data(self) -> pd.DataFrame:
        """입력 데이터 로드"""
        df = pd.read_excel(
            self.config.input_excel_path,
            sheet_name=self.config.input_sheet_name,
            engine="openpyxl"
        )

        required_cols = [
            self.config.input_name_col,
            self.config.input_code_col,
            self.config.input_asset_col,
            self.config.input_date_col
        ]

        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"'{col}' 컬럼이 없습니다. 현재 컬럼: {list(df.columns)}")

        # 정리
        df = df[required_cols].copy()
        df[self.config.input_name_col] = df[self.config.input_name_col].astype(str).str.strip()
        df = df[df[self.config.input_name_col] != ""]
        df = df.drop_duplicates(subset=[self.config.input_name_col], keep="first")

        return df

    def classify_with_llm(self, names: List[str]) -> None:
        """LLM으로 분류 (캐시 활용)"""
        todo = [n for n in names if not self.cache.contains(n)]

        if not todo:
            print(f"[INFO] 모든 상품이 캐시에 있습니다.")
            return

        print(f"[INFO] 분류할 상품: {len(todo)}개")

        for i in range(0, len(todo), self.config.batch_size):
            batch = todo[i:i + self.config.batch_size]
            print(f"[INFO] 분류 중 {i + 1}~{i + len(batch)} / {len(todo)}")

            try:
                results = self.llm_client.classify(batch)
                self.cache.save_batch(results)
            except Exception as e:
                print(f"[WARN] 배치 실패: {e}")

            time.sleep(self.config.rate_limit_delay)

    def build_labeled_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """라벨링된 DataFrame 생성"""
        name_col = self.config.input_name_col
        names = df[name_col].tolist()

        records = []
        for name in names:
            cached = self.cache.get(name)
            if cached:
                records.append({
                    name_col: name,
                    "country_free": cached.get("country", "기타"),
                    "asset_type_free": cached.get("asset_type", "기타"),
                    "theme_free": cached.get("theme", "기타")
                })
            else:
                records.append({
                    name_col: name,
                    "country_free": "기타",
                    "asset_type_free": "기타",
                    "theme_free": "기타"
                })

        labeled = pd.DataFrame(records)
        labeled = labeled.merge(df, on=name_col, how="left")

        return labeled

    def normalize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """라벨 정규화"""
        df["country_norm"] = df["country_free"].apply(self.normalizer.normalize_country)
        df["asset_type_norm"] = df["asset_type_free"].apply(self.normalizer.normalize_asset_type)
        df["theme_norm"] = df["theme_free"].apply(self.normalizer.normalize_theme)
        return df

    def refine_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """빈도 기반 정제"""
        df["country_final"] = self.country_refiner.refine(df["country_norm"])
        df["asset_type_final"] = df["asset_type_norm"]  # 3개뿐이므로 정제 불필요
        df["theme_final"] = self.theme_refiner.refine(df["theme_norm"])
        df["factor_key_final"] = (
            df["country_final"] + "-" +
            df["asset_type_final"] + "-" +
            df["theme_final"]
        )
        return df

    def compute_frequencies(self, df: pd.DataFrame) -> pd.DataFrame:
        """빈도 테이블 생성"""
        freq_parts = []

        for col, axis_name in [
            ("country_free", "country_free"),
            ("asset_type_free", "asset_type_free"),
            ("theme_free", "theme_free"),
            ("country_norm", "country_norm"),
            ("asset_type_norm", "asset_type_norm"),
            ("theme_norm", "theme_norm"),
        ]:
            freq = df[col].value_counts().rename_axis("label").reset_index(name="count")
            freq["axis"] = axis_name
            freq_parts.append(freq)

        return pd.concat(freq_parts, ignore_index=True)[["axis", "label", "count"]]

    def compute_cross_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """교차 분포 계산"""
        cross = (
            df.groupby(["country_final", "asset_type_final", "theme_final"])
            .size()
            .reset_index(name="count")
            .sort_values(
                ["country_final", "asset_type_final", "theme_final", "count"],
                ascending=[True, True, True, False]
            )
        )
        return cross

    def save_results(
        self,
        labeled_free: pd.DataFrame,
        labeled_final: pd.DataFrame,
        frequencies: pd.DataFrame,
        cross_counts: pd.DataFrame
    ) -> None:
        """결과 저장"""
        import os

        # 출력 폴더 생성
        os.makedirs(self.config.output_dir, exist_ok=True)

        # CSV 파일 저장
        labeled_free.to_csv(
            self.config.output_labeled_free,
            index=False,
            encoding="utf-8-sig"
        )
        labeled_final.to_csv(
            self.config.output_labeled_final,
            index=False,
            encoding="utf-8-sig"
        )
        frequencies.to_csv(
            self.config.output_frequencies,
            index=False,
            encoding="utf-8-sig"
        )
        cross_counts.to_csv(
            self.config.output_cross_counts,
            index=False,
            encoding="utf-8-sig"
        )

        print(f"\n[완료] 자유 라벨: {self.config.output_labeled_free}")
        print(f"[완료] 최종 라벨: {self.config.output_labeled_final}")
        print(f"[완료] 빈도표: {self.config.output_frequencies}")
        print(f"[완료] 교차분포: {self.config.output_cross_counts}")

        # 상품명.xlsx에 분류 결과 추가
        self.update_product_name_file(labeled_final)

    def update_product_name_file(self, labeled_final: pd.DataFrame) -> None:
        """상품명.xlsx 파일에 분류 결과 컬럼 추가"""
        product_file = self.config.product_name_file

        # 상품명 파일 로드
        product_df = pd.read_excel(product_file)
        print(f"\n[상품명 파일 업데이트]")
        print(f"  원본: {len(product_df)}행")

        # 분류 결과에서 필요한 컬럼만 추출 (코드, 지역, 주식채권구분, 테마)
        merge_cols = labeled_final[[
            self.config.input_code_col,
            "country_final",
            "asset_type_final",
            "theme_final"
        ]].copy()

        # 컬럼명 변경
        merge_cols = merge_cols.rename(columns={
            "country_final": "지역",
            "asset_type_final": "주식채권구분",
            "theme_final": "테마"
        })

        # 기존 컬럼 제거 (있으면)
        for col in ["지역", "주식채권구분", "테마"]:
            if col in product_df.columns:
                product_df = product_df.drop(columns=[col])

        # 병합
        product_df = product_df.merge(
            merge_cols,
            on=self.config.input_code_col,
            how="left"
        )

        # 저장
        product_df.to_excel(product_file, index=False)
        print(f"  업데이트 완료: {product_file}")
        print(f"  추가된 컬럼: 지역, 주식채권구분, 테마")

    def run(self) -> None:
        """전체 파이프라인 실행"""
        import os

        # 출력 폴더 생성 (캐시 파일 저장을 위해 먼저 생성)
        os.makedirs(self.config.output_dir, exist_ok=True)

        print("=" * 70)
        print("[1] 데이터 로드")
        print("=" * 70)
        df = self.load_data()
        names = df[self.config.input_name_col].tolist()
        print(f"  유니크 상품명: {len(names)}개")

        print("\n" + "=" * 70)
        print("[2] LLM 분류")
        print("=" * 70)
        self.classify_with_llm(names)

        print("\n" + "=" * 70)
        print("[3] 라벨 DataFrame 생성")
        print("=" * 70)
        labeled_free = self.build_labeled_dataframe(df)
        print(f"  생성 완료: {len(labeled_free)}행")

        print("\n" + "=" * 70)
        print("[4] 라벨 정규화")
        print("=" * 70)
        labeled_free = self.normalize_labels(labeled_free)
        print("  정규화 완료")

        print("\n" + "=" * 70)
        print("[5] 빈도 기반 정제")
        print("=" * 70)
        labeled_final = self.refine_labels(labeled_free.copy())
        print("  정제 완료")

        print("\n" + "=" * 70)
        print("[6] 통계 계산")
        print("=" * 70)
        frequencies = self.compute_frequencies(labeled_free)
        cross_counts = self.compute_cross_counts(labeled_final)
        print("  계산 완료")

        print("\n" + "=" * 70)
        print("[7] 결과 저장")
        print("=" * 70)
        self.save_results(labeled_free, labeled_final, frequencies, cross_counts)

        # 요약 출력
        print("\n" + "=" * 70)
        print("요약")
        print("=" * 70)
        print("\n[국가 분포 (정규화 후)]")
        print(labeled_final["country_final"].value_counts().head(10).to_string())
        print("\n[자산유형 분포]")
        print(labeled_final["asset_type_final"].value_counts().to_string())
        print("\n[테마 분포 (정규화 후 상위 10)]")
        print(labeled_final["theme_final"].value_counts().head(10).to_string())


# =============================================================================
# 메인 실행
# =============================================================================

if __name__ == "__main__":
    config = LLMConfig()
    classifier = ProductClassifier(config)
    classifier.run()
