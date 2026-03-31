"""
extractor.py — LLM 기반 논증 추출

기사 전체를 GPT에게 읽히고, 토론에 사용할 수 있는 논증을 직접 추출한다.
기계적 청킹 + 별도 라벨링 대신, 한 번의 LLM 호출로 고품질 논증을 생성한다.

[기존 방식]
    기사 → 문단 분할 (chunker) → 라벨링 (labeler)  ← 노이즈에 취약

[개선 방식]
    기사 → GPT가 통째로 읽고 논증 추출 (extractor)  ← 노이즈 자동 무시

[비용]
    기사당 GPT-4o-mini 1회 호출 ≈ $0.0003
    12토픽 × 20기사 = 240회 ≈ $0.07
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Dict, List, Optional

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

logger = logging.getLogger(__name__)

_extractor_llm = None


def _get_llm() -> ChatOpenAI:
    global _extractor_llm
    if _extractor_llm is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY 환경변수가 설정되지 않았습니다. "
                "export OPENAI_API_KEY=sk-... 로 설정하세요."
            )
        _extractor_llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.2,
            model_kwargs={"response_format": {"type": "json_object"}},
        )
    return _extractor_llm


VALID_STANCES = {"PRO", "CON", "NEUTRAL"}
VALID_CLAIM_TYPES = {
    "claim", "evidence", "statistic",
    "example", "counterargument", "expert_opinion",
}


def extract_arguments(
    article_text: str,
    topic: dict,
    max_arguments: int = 3,
) -> List[Dict]:
    """기사 전체에서 토론 논증을 추출한다.

    GPT가 기사를 통째로 읽고, 토론에 사용할 수 있는 논증을 직접 추출한다.
    네비게이션 메뉴, 광고, 관련 없는 내용은 자동으로 무시된다.

    Args:
        article_text: 전처리된 기사 전체 텍스트
        topic:        topics.json 항목 (title, pro, con 포함)
        max_arguments: 기사당 최대 추출 논증 수

    Returns:
        추출된 논증 리스트. 각 항목:
        {"text", "stance", "stance_score", "claim_type", "relevance_score"}
        관련 없는 기사면 빈 리스트 반환.
    """
    title = topic["title"]
    pro = topic.get("pro", "")
    con = topic.get("con", "")

    # 입력 텍스트가 너무 길면 앞부분만 사용 (GPT-4o-mini 컨텍스트 제한 대응)
    max_input_chars = 12000
    if len(article_text) > max_input_chars:
        article_text = article_text[:max_input_chars] + "\n\n[이하 생략]"

    prompt = f"""다음 기사를 읽고, 아래 토론 논제에 사용할 수 있는 논증(argument)을 최대 {max_arguments}개 추출하세요.

[토론 논제]
{title}
- 찬성(PRO): {pro}
- 반대(CON): {con}

[기사]
{article_text}

[지시사항]
1. 기사에서 위 토론 논제와 관련된 핵심 논증을 추출하세요.
2. 각 논증은 "주장 + 근거"가 포함된 150~400자의 독립적인 텍스트여야 합니다.
3. 구체적인 수치, 기관명, 연구 결과를 반드시 포함하세요.
4. 토론자가 직접 인용할 수 있는 단정적 문체로 작성하세요. "~로 분석된다", "~로 보인다" 같은 간접 서술 대신, "~이다", "~을 보여준다", "~에 따르면" 형태를 사용하세요.
5. PRO와 CON 논증을 균형 있게 추출하세요. 기사가 한쪽 입장이어도, 반대 입장의 반론 근거가 될 수 있는 내용을 찾아 CON(또는 PRO)으로 추출하세요.
6. 네비게이션 메뉴, 광고, 저자 소개 등은 무시하세요.
7. 기사가 토론 논제와 관련이 없으면 빈 배열 []을 반환하세요.
8. 서로 의미적으로 중복되는 논증은 추출하지 마세요. 각 논증은 다른 각도나 다른 근거를 다뤄야 합니다.

[출력 형식]
{{"arguments": [
  {{
    "text": "추출한 논증 텍스트 (150~400자)",
    "stance": "PRO 또는 CON 또는 NEUTRAL",
    "stance_score": -1.0에서 1.0 사이 실수,
    "claim_type": "claim/evidence/statistic/example/counterargument/expert_opinion",
    "relevance_score": 0.0에서 1.0 사이 실수
  }}
]}}

claim_type 기준:
- claim: 명시적 주장·입장 표명
- evidence: 사실 기반 근거
- statistic: 수치·통계 데이터
- example: 실제 사례·케이스
- counterargument: 상대 논리에 대한 반박
- expert_opinion: 전문가·기관 의견 인용"""

    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = _get_llm()
        messages = [
            SystemMessage(content="You are a debate research assistant. Always respond in JSON format."),
            HumanMessage(content=prompt),
        ]
        response = llm.invoke(messages)
        content = response.content if isinstance(response.content, str) else str(response.content)

        # JSON 파싱 (3단계 폴백)
        data = _parse_json(content)
        if not data or "arguments" not in data:
            logger.warning("[extractor] JSON 파싱 실패 또는 arguments 키 없음")
            return []

        # 각 논증 검증
        results = []
        for arg in data["arguments"]:
            validated = _validate_argument(arg)
            if validated:
                results.append(validated)

        return results[:max_arguments]

    except Exception as e:
        logger.warning("[extractor] 논증 추출 실패: %s", e)
        return []


def _parse_json(content: str) -> Optional[dict]:
    """LLM 응답에서 JSON을 추출한다."""
    text = content.strip()

    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def _validate_argument(arg: dict) -> Optional[Dict]:
    """추출된 논증의 필드를 검증하고 보정한다."""
    text = arg.get("text", "").strip()
    if len(text) < 50:
        return None

    stance = arg.get("stance", "NEUTRAL")
    if stance not in VALID_STANCES:
        stance = "NEUTRAL"

    stance_score = float(arg.get("stance_score", 0.0))
    stance_score = max(-1.0, min(1.0, stance_score))

    if abs(stance_score) < 0.2:
        stance = "NEUTRAL"

    claim_type = arg.get("claim_type", "evidence")
    if claim_type not in VALID_CLAIM_TYPES:
        claim_type = "evidence"

    relevance_score = float(arg.get("relevance_score", 0.5))
    relevance_score = max(0.0, min(1.0, relevance_score))

    if relevance_score < 0.4:
        return None

    return {
        "text": text,
        "stance": stance,
        "stance_score": round(stance_score, 2),
        "claim_type": claim_type,
        "relevance_score": round(relevance_score, 2),
    }
