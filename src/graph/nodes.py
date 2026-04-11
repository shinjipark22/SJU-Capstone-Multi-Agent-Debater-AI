"""
nodes.py — 서브그래프 노드 함수 (search, write, review)

LangGraph StateGraph 노드로 사용됨.
각 함수는 TurnState를 받아 부분 업데이트 dict를 반환.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.graph.searcher import (
    analyze_weakness,
    decide_search,
    generate_attack_question,
    search_web,
)
from src.graph.writer import (
    invoke_with_retry,
    truncate_tool_result,
    main_llm,
    rebuttal_llm,
    free_rebuttal_llm,
    synthesis_llm,
    role_reversal_llm,
)
from src.graph.reviewer import review_speech
from src.graph.turn_state import TurnState

logger = logging.getLogger(__name__)


# ── 텍스트 추출/후처리 (기존 stage1/nodes.py에서 임포트) ────────────────────
from src.stage1_opening.nodes import (
    _postprocess_speech,
    _extract_delimited_text,
    _is_valid_speech,
)
from src.stage2_rebuttal.nodes import (
    _extract_rebuttal_text,
    _is_valid_rebuttal,
)


# ── Search 노드 ───────────────────────────────────────────────────────────

def search_node(state: TurnState) -> dict:
    """Searcher: 약점 분석 + 검색 판단 + 검색 실행."""
    tool_calls_log = list(state.get("tool_calls_log", []))

    # 약점 분석
    weakness = analyze_weakness(
        state["target_argument"],
        state["topic"],
        prev_weaknesses=state.get("prev_weaknesses", ""),
    )
    if weakness:
        tool_calls_log.append({"name": "analyze_weakness", "result": weakness})

    # 검색 판단
    query = decide_search(state["target_argument"])
    search_results = ""
    if query:
        tool_calls_log.append({"name": "search_web", "args": {"query": query}})
        web_result = search_web.invoke({"query": query})
        search_results = truncate_tool_result(web_result)

    return {
        "weakness": weakness,
        "search_results": search_results,
        "search_query": query,
        "tool_calls_log": tool_calls_log,
    }


# ── Write 노드 ────────────────────────────────────────────────────────────

def _select_llm(mode: str):
    """모드에 따라 적절한 LLM 선택."""
    if mode == "opening":
        return main_llm
    elif mode == "rebuttal":
        return rebuttal_llm
    elif mode in ("defense", "attack"):
        return free_rebuttal_llm
    elif mode == "role_reversal":
        return role_reversal_llm
    elif mode == "synthesis":
        return synthesis_llm
    return main_llm


def write_node(state: TurnState) -> dict:
    """Writer: 발언 생성. chain + 프롬프트로 LLM 호출."""
    mode = state.get("mode", "attack")
    llm = _select_llm(mode)

    agent = state["agent"]
    stance_kr = "찬성" if state["expected_stance"] == "PRO" else "반대"

    # 시스템 메시지 구성
    system = (
        f"{agent['system_prompt']}\n\n"
        f"[최우선 규칙] 너는 {stance_kr} 입장이다. "
        f"반드시 3~4문장으로만 답변하라. "
        f"반드시 한국어로만 답변하라."
    )

    # 메시지 체인 구성
    messages = [SystemMessage(content=system)]
    chain = state.get("chain", [])
    messages.extend(chain)

    # 프롬프트 구성 (모드별)
    weakness = state.get("weakness", "")
    search_results = state.get("search_results", "")
    target = state.get("target_argument", "")
    prev_attacks = state.get("prev_attacks", "")

    if mode == "defense":
        prompt = _build_defense_prompt(target, state.get("my_opening", ""), search_results)
    elif mode == "attack":
        prompt = _build_attack_prompt(target, weakness, search_results, state.get("opp_opening", ""), prev_attacks)
    elif mode == "opening":
        prompt = target  # 입론은 외부에서 프롬프트를 완성하여 전달
    elif mode == "synthesis":
        prompt = target  # 종합도 외부에서 프롬프트 전달
    else:
        prompt = target  # role_reversal 등

    messages.append(HumanMessage(content=prompt))

    # LLM 호출
    response = invoke_with_retry(llm, messages, label=mode)
    raw = response.content if isinstance(response.content, str) else str(response.content)

    # 후처리
    if mode == "opening" or mode == "role_reversal":
        speech = _postprocess_speech(_extract_delimited_text(raw))
    else:
        speech = _postprocess_speech(_extract_rebuttal_text(raw))

    return {"speech": speech, "raw": raw}


# ── Review 노드 ───────────────────────────────────────────────────────────

def review_node(state: TurnState) -> dict:
    """Reviewer: 품질 검증."""
    speech = state.get("speech", "")
    expected_stance = state.get("expected_stance", "PRO")
    topic = state.get("topic", "")
    mode = state.get("mode", "attack")

    # 종합 회의에서는 stance 체크 안 함 (중립이어야 하니까)
    check_stance = mode not in ("synthesis",)

    result = review_speech(speech, expected_stance, topic, check_stance_flag=check_stance)

    retry_count = state.get("retry_count", 0) + 1

    return {"review_result": result, "retry_count": retry_count}


# ── 라우터 ─────────────────────────────────────────────────────────────────

def route_review(state: TurnState) -> str:
    """조건부 라우팅: 통과 or 재시도."""
    result = state.get("review_result", {})
    retry_count = state.get("retry_count", 0)

    if result.get("passed", True):
        return "pass"
    if retry_count >= 3:
        logger.warning("[router] 3회 재시도 초과, 현재 발언 사용")
        return "pass"
    return "retry"


# ── 공격/방어 프롬프트 빌더 ────────────────────────────────────────────────

def _build_defense_prompt(user_attack: str, my_opening: str, search_results: str) -> str:
    my_block = f"\n[나의 입론 — 이것을 근거로 방어하라]\n{my_opening[:300]}\n" if my_opening else ""
    ref_block = f"\n[참고 자료]\n{search_results}\n" if search_results else ""

    return f"""상대의 공격:
{user_attack}
{my_block}{ref_block}
상대가 너의 주장의 허점을 지적하고 있다.
- 상대가 질문했으면 그 질문에 먼저 직접 답하라
- 나의 입론이나 참고 자료에서 근거를 찾아 구체적으로 반박하라
- 이전 턴에서 이미 사용한 데이터/출처/문구를 재사용하지 마라. 새로운 근거로 방어하라

[규칙]
- 3~4문장으로만 답변
- 핵심에 **강조** 사용
- 합니다체(격식체)
- 한국어로 작성. 고유명사만 영어 허용

반드시 아래 형식으로만 출력:

### 반박 시작
(반박 내용)
### 반박 끝"""


def _build_attack_prompt(
    target: str, weakness: str, search_results: str,
    opp_opening: str, prev_attacks: str,
) -> str:
    ref_block = f"\n[참고 자료 — 반박 근거로 활용하라]\n{search_results}\n" if search_results else ""
    weakness_hint = f"\n[약점 분석 — 이 부분을 집중 공격하라]\n{weakness}\n" if weakness else ""
    opp_block = f"\n[상대 입론 — 지금 발언과 모순되는 부분이 있으면 지적하라]\n{opp_opening[:300]}\n" if opp_opening else ""
    prev_hint = f"\n[이전 공격 — 아래 내용은 이미 사용했으니 반복 금지]\n{prev_attacks}\n" if prev_attacks else ""

    return f"""상대 발언:
{target}
{ref_block}{weakness_hint}{opp_block}{prev_hint}
상대 발언에서 논리적 허점, 근거 부족, 과장된 주장을 찾아 공격하라.
- 상대가 인용한 수치/출처의 신뢰성을 검증하라
- 이전 턴에서 이미 사용한 데이터/출처/문구를 재사용하지 마라. 매 턴 새로운 근거나 관점으로 공격하라

[규칙]
- 3~4문장으로만 답변
- 핵심에 **강조** 사용
- 합니다체(격식체)
- 한국어로 작성. 고유명사만 영어 허용
- 참고 자료의 수치만 인용 가능. 자체적으로 수치를 지어내지 마라

반드시 아래 형식으로만 출력:

### 반박 시작
(반박 내용)
### 반박 끝"""
