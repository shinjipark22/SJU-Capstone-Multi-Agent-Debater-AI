"""
labeler.py — LLM 기반 메타데이터 라벨링

각 청크에 stance, stance_score, claim_type을 GPT-4o-mini 기반으로 자동 부여한다.
일회성 전처리이므로 외부 API를 사용하여 JSON 출력 안정성과 라벨링 품질을 확보한다.

[라벨 필드]
    - stance: PRO | CON | NEUTRAL
    - stance_score: -1.0 ~ +1.0 (PRO=양수, CON=음수)
    - claim_type: claim | evidence | statistic | example | counterargument | expert_opinion

[비용]
    GPT-4o-mini 기준 240건 ≈ $0.3 이하
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()  # 프로젝트 루트의 .env 파일에서 OPENAI_API_KEY 로드

logger = logging.getLogger(__name__)

_labeler_llm = None  # lazy init


def _get_llm() -> ChatOpenAI:
    global _labeler_llm
    if _labeler_llm is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 환경변수가 설정되지 않았습니다. "
                "export OPENAI_API_KEY=sk-... 로 설정하세요."
            )
        _labeler_llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.2,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _labeler_llm


VALID_STANCES = {"PRO", "CON", "NEUTRAL"}
VALID_CLAIM_TYPES = {
    "claim", "evidence", "statistic",
    "example", "counterargument", "expert_opinion",
}


def _parse_llm_json(content: str) -> Optional[dict]:
    """LLM 응답에서 JSON을 추출한다 (3단계 폴백)."""
    # 1. <think> 블록 제거
    text = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = text.replace('</think>', '').strip()

    # 2. 직접 파싱
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. 정규식으로 JSON 블록 탐색
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def label_chunk(chunk: str, topic: dict) -> Optional[Dict]:
    """단일 청크에 stance, stance_score, claim_type 라벨을 부여한다.

    Args:
        chunk: 라벨링할 텍스트 청크
        topic: topics.json 항목 (title, pro, con 포함)

    Returns:
        {"stance", "stance_score", "claim_type", "relevance_score"} 또는 실패/저품질 시 None
    """
    title = topic["title"]
    pro = topic.get("pro", "")
    con = topic.get("con", "")

    prompt = f"""다음 텍스트를 분석하여 JSON으로 출력하세요.

[토론 논제]
{title}
- 찬성(PRO): {pro}
- 반대(CON): {con}

[텍스트]
{chunk}

[출력 형식]
{{"stance": "PRO 또는 CON 또는 NEUTRAL", "stance_score": -1.0에서 1.0 사이 실수, "claim_type": "아래 중 하나", "relevance_score": 0.0에서 1.0 사이 실수}}

판단 기준:
- stance: 텍스트가 찬성 주장을 지지하면 PRO, 반대 주장을 지지하면 CON, 어느 쪽도 아니면 NEUTRAL
- stance_score: 강한 찬성 +0.8~1.0 / 약한 찬성 +0.3~0.5 / 중립 ±0.2 이내 / 약한 반대 -0.3~-0.5 / 강한 반대 -0.8~-1.0
- claim_type:
  claim = 명시적 주장·입장 표명
  evidence = 사실 기반 근거
  statistic = 수치·통계 데이터
  example = 실제 사례·케이스
  counterargument = 상대 논리에 대한 반박
  expert_opinion = 전문가·기관 의견 인용
- relevance_score: 이 텍스트가 위 토론 논제에 얼마나 관련 있는지 평가
  1.0 = 논제를 직접 다루는 핵심 근거
  0.7~0.9 = 논제와 밀접하게 관련된 배경/사례
  0.4~0.6 = 간접적으로 관련 있음
  0.0~0.3 = 관련 없거나 광고/잡음"""

    try:
        llm = _get_llm()
        response = llm.invoke(prompt)
        content = response.content if isinstance(response.content, str) else str(response.content)

        data = _parse_llm_json(content)
        if not data:
            logger.warning("[labeler] JSON 파싱 실패")
            return None

        # ── 값 검증 + 보정 ────────────────────────────────────────────────
        stance = data.get("stance", "NEUTRAL")
        if stance not in VALID_STANCES:
            stance = "NEUTRAL"

        stance_score = float(data.get("stance_score", 0.0))
        stance_score = max(-1.0, min(1.0, stance_score))

        # confidence가 낮으면 NEUTRAL로 보정
        if abs(stance_score) < 0.2:
            stance = "NEUTRAL"

        claim_type = data.get("claim_type", "evidence")
        if claim_type not in VALID_CLAIM_TYPES:
            claim_type = "evidence"

        relevance_score = float(data.get("relevance_score", 0.5))
        relevance_score = max(0.0, min(1.0, relevance_score))

        # 품질 필터: relevance_score가 0.4 미만이면 버린다
        if relevance_score < 0.4:
            logger.info("[labeler] 저품질 청크 필터링 (relevance=%.2f)", relevance_score)
            return None

        return {
            "stance": stance,
            "stance_score": round(stance_score, 2),
            "claim_type": claim_type,
            "relevance_score": round(relevance_score, 2),
        }

    except Exception as e:
        logger.warning("[labeler] 라벨링 실패: %s", e)
        return None


def label_chunks_batch(chunks: List[str], topic: dict) -> List[Optional[Dict]]:
    """여러 청크를 순차적으로 라벨링한다."""
    return [label_chunk(chunk, topic) for chunk in chunks]
