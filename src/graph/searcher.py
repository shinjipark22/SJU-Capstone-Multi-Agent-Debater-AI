"""
searcher.py — Qwen 7B 기반 분석/검색 모듈

역할:
    - 약점 분석 (analyze_weakness)
    - 검색 필요 여부 판단 (decide_search)
    - 공격 질문 생성 (generate_attack_question)
    - 입장 검증 (check_stance)
    - 웹 검색 (search_web)
"""

from __future__ import annotations

import logging
import os
import re
from typing import List

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

logger = logging.getLogger(__name__)

# ── .env 로드 ──────────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            if _line.strip().startswith("TAVILY_API_KEY="):
                os.environ["TAVILY_API_KEY"] = _line.strip().split("=", 1)[1]
                break

# ── Qwen 7B vLLM 클라이언트 ────────────────────────────────────────────────
_QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:8001/v1")
qwen_llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url=_QWEN_BASE_URL,
    api_key="fake",
    temperature=0.3,
    max_tokens=200,
    timeout=30,
)

# ── Tavily 검색 클라이언트 ──────────────────────────────────────────────────
_tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))


# ── 웹 검색 ────────────────────────────────────────────────────────────────

@tool
def search_web(query: str) -> str:
    """웹에서 최신 뉴스 및 정보를 검색합니다."""
    try:
        results = _tavily_client.search(
            query,
            max_results=3,
            search_depth="basic",
            exclude_domains=[
                "blog.naver.com", "m.blog.naver.com",
                "tistory.com", "brunch.co.kr",
                "linkedin.com", "medium.com",
                "velog.io", "daum.net",
            ],
        )
        items = results.get("results", [])
        if not items:
            return "[검색 결과] 관련 결과를 찾을 수 없습니다."
        return "[검색 결과]\n" + "\n".join(
            f"- {r['title']}: {r['content'][:200]}" for r in items
        )
    except Exception as e:
        return f"[검색 오류] {e}"


# ── 검색 필요 여부 판단 ────────────────────────────────────────────────────

def decide_search(target_argument: str) -> str:
    """검색 필요 여부를 판단한다. 필요하면 검색 키워드, 불필요하면 빈 문자열."""
    try:
        messages = [
            HumanMessage(content=f"""다음 주장을 반박하려 한다. 반박에 통계나 사실 확인이 필요하면 검색 키워드를 한국어 30자 이내로 출력하라.
논리만으로 반박 가능하면 "불필요"라고만 출력하라.

주장: {target_argument[:200]}""")
        ]
        response = qwen_llm.invoke(messages)
        result = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()

        if "불필요" in result or len(result) < 3:
            logger.info("[searcher] 검색 불필요")
            return ""

        query = result.split('\n')[0].strip().strip('"').strip("'")
        logger.info("[searcher] 검색 키워드: '%s'", query[:40])
        return query[:40]
    except Exception as e:
        logger.warning("[searcher] decide_search 오류: %s", e)
        return ""


# ── 약점 분석 ──────────────────────────────────────────────────────────────

def analyze_weakness(target_speech: str, topic: str, prev_weaknesses: str = "") -> str:
    """상대 논거의 핵심 약점을 분석한다. 이전 분석과 다른 약점을 찾는다."""
    prev_block = ""
    if prev_weaknesses:
        prev_block = f"\n[이미 분석한 약점 — 이것과 다른 새로운 약점을 찾아라]\n{prev_weaknesses}\n"

    try:
        messages = [
            HumanMessage(content=f"""다음 주장의 가장 약한 부분을 1줄로 짚어라.

토론 주제: {topic[:80]}
상대 주장: {target_speech[:300]}
{prev_block}
[필수] 반드시 한국어로만 답변하라. 영어, 중국어 등 다른 언어 사용 금지.
{f"이전에 분석한 약점과 완전히 다른 관점의 약점을 찾아라." if prev_weaknesses else ""}
형식: "약점: (내용)" 한 줄만 출력.""")
        ]
        response = qwen_llm.invoke(messages)
        result = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
        # 중국어 유출 필터링
        if re.search(r'[\u4e00-\u9fff]', result):
            logger.warning("[searcher] 중국어 유출 감지, 결과 폐기")
            return ""
        if "약점:" in result:
            return result.split("약점:")[-1].strip()
        return result.split('\n')[0].strip()
    except Exception as e:
        logger.warning("[searcher] analyze_weakness 오류: %s", e)
        return ""


# ── 공격 질문 생성 ─────────────────────────────────────────────────────────

def generate_attack_question(attack_text: str, stance: str, topic: str) -> str:
    """공격 발언의 흐름에 맞는 마무리 질문을 생성한다."""
    try:
        messages = [
            HumanMessage(content=f"""다음 공격 발언을 읽고, 이 흐름에 맞는 마무리 질문을 1개 만들어라.

공격 발언: {attack_text[:300]}

규칙:
- 공격 내용과 자연스럽게 이어지는 질문
- "~할 수 있습니까?", "~라고 보십니까?", "~지 않습니까?" 형태
- 반드시 한국어로만 출력. 영어/중국어 등 다른 언어 금지
- 합니다체
- 한 문장만 출력. ?로 끝나야 함
- 설명하지 말고 질문만 출력

예시: 그렇다면 상대는 이 문제를 어떻게 해결할 수 있다고 보십니까?""")
        ]
        response = qwen_llm.invoke(messages)
        result = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
        for line in result.split('\n'):
            line = line.strip()
            if line.endswith('?') and re.search(r'[가-힣]', line):
                return line
        return ""
    except Exception as e:
        logger.warning("[searcher] generate_attack_question 오류: %s", e)
        return ""


# ── 입장 검증 ──────────────────────────────────────────────────────────────

def check_stance(text: str, expected_stance: str, topic: str) -> bool:
    """발언이 기대 입장과 일치하는지 판별한다. 일치하면 True."""
    stance_kr = "찬성" if expected_stance == "PRO" else "반대"
    try:
        messages = [
            HumanMessage(content=f"""주제: {topic[:100]}

발언: {text[:200]}

이 발언은 위 주제에 대해 "찬성"인가 "반대"인가? 한 단어로만 답하라.""")
        ]
        response = qwen_llm.invoke(messages)
        result = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
        first_word = result.split()[0] if result.split() else ""
        detected = "찬성" if "찬성" in first_word else ("반대" if "반대" in first_word else "")
        if detected and detected != stance_kr:
            logger.warning("[searcher] 입장 혼동: 기대=%s, 감지=%s", stance_kr, detected)
            return False
        return True
    except Exception as e:
        logger.warning("[searcher] check_stance 오류: %s", e)
        return True
