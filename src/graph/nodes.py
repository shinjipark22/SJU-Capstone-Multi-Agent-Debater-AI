"""
nodes.py — 서브그래프 노드 함수 (search, write, review)

LangGraph StateGraph 노드로 사용됨.
각 함수는 TurnState를 받아 부분 업데이트 dict를 반환.

[중요] write_node는 이전 stage별 세밀한 재시도/영어감지/후처리를
       그대로 보존한다. 범용 로직으로 대체하지 않는다.
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


# ── 텍스트 추출/후처리 (기존 stage 로직 그대로) ────────────────────────────
from src.phase1.stage1_opening.nodes import (
    _postprocess_speech,
    _extract_delimited_text,
    _is_valid_speech,
)
from src.phase1.stage2_rebuttal.nodes import (
    _extract_rebuttal_text,
    _is_valid_rebuttal,
)


# ── Search 노드 ───────────────────────────────────────────────────────────

def search_node(state: TurnState) -> dict:
    """Searcher: 약점 분석 + 검색 판단 + 검색 실행."""
    tool_calls_log = list(state.get("tool_calls_log", []))

    weakness = analyze_weakness(
        state["target_argument"],
        state["topic"],
        prev_weaknesses=state.get("prev_weaknesses", ""),
    )
    if weakness:
        tool_calls_log.append({"name": "analyze_weakness", "result": weakness})

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
# 각 mode별로 이전 stage의 생성+재시도+영어감지 로직을 그대로 보존

def _select_llm(mode: str):
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


def _build_defense_prompt(user_attack, my_opening, search_results):
    my_block = f"\n[나의 입론 — 이것을 근거로 방어하라]\n{my_opening[:300]}\n" if my_opening else ""
    ref_block = f"\n[참고 자료]\n{search_results}\n" if search_results else ""
    return f"""상대의 공격:
{user_attack}
{my_block}{ref_block}
상대가 너의 주장의 허점을 지적하고 있다.
- 상대가 질문했으면 그 질문에 먼저 직접 답하라
- 나의 입론이나 참고 자료에서 근거를 찾아 구체적으로 반박하라
- 이전 턴에서 이미 사용한 데이터/출처/문구를 재사용하지 마라

[규칙]
- 3~4문장으로만 답변
- 핵심에 **강조** 사용
- 합니다체(격식체)
- 한국어로 작성. 고유명사만 영어 허용

반드시 아래 형식으로만 출력:

### 반박 시작
(반박 내용)
### 반박 끝"""


def _build_attack_prompt(target, weakness, search_results, opp_opening, prev_attacks):
    ref_block = f"\n[참고 자료 — 반박 근거로 활용하라]\n{search_results}\n" if search_results else ""
    weakness_hint = f"\n[약점 분석 — 이 부분을 집중 공격하라]\n{weakness}\n" if weakness else ""
    opp_block = f"\n[상대 입론 — 모순 지적]\n{opp_opening[:300]}\n" if opp_opening else ""
    prev_hint = f"\n[이전 공격 — 반복 금지]\n{prev_attacks}\n" if prev_attacks else ""
    return f"""상대 발언:
{target}
{ref_block}{weakness_hint}{opp_block}{prev_hint}
상대 발언에서 논리적 허점, 근거 부족, 과장된 주장을 찾아 공격하라.
- 이전 턴에서 이미 사용한 데이터/출처/문구를 재사용하지 마라

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


def write_node(state: TurnState) -> dict:
    """Writer: 발언 생성.

    각 mode별로 이전 stage의 재시도+영어감지+후처리 로직을 그대로 보존.
    """
    mode = state.get("mode", "attack")
    llm = _select_llm(mode)
    agent = state["agent"]
    stance_kr = "찬성" if state["expected_stance"] == "PRO" else "반대"
    target = state.get("target_argument", "")
    weakness = state.get("weakness", "")
    search_results = state.get("search_results", "")

    # ── 시스템 메시지
    system = agent["system_prompt"]
    if mode in ("defense", "attack"):
        system += (
            f"\n\n[최우선 규칙] 너는 {stance_kr} 입장이다. "
            f"반드시 3~4문장으로만 답변하라. "
            f"상대 주장의 오류만 공격하라."
        )
    elif mode == "synthesis":
        system += (
            f"\n\n[최우선 규칙] 지금은 최적해 회의 중이다. "
            f"찬성/반대 입장을 완전히 버려라. "
            f"중립적 관점에서 최선의 해결책을 함께 찾아라. "
            f"1~2문장으로만 답변하라. 반드시 한국어로만 답변하라."
        )

    # ── 메시지 구성
    messages = [SystemMessage(content=system)]
    chain = state.get("chain", [])
    messages.extend(chain)

    # ── 프롬프트
    if mode == "defense":
        prompt = _build_defense_prompt(target, state.get("my_opening", ""), search_results)
    elif mode == "attack":
        prompt = _build_attack_prompt(
            target, weakness, search_results,
            state.get("opp_opening", ""), state.get("prev_attacks", ""),
        )
    else:
        prompt = target  # opening, rebuttal, role_reversal, synthesis는 외부에서 프롬프트 완성

    messages.append(HumanMessage(content=prompt))

    # ── LLM 호출
    response = invoke_with_retry(llm, messages, label=mode)
    raw = response.content if isinstance(response.content, str) else str(response.content)

    # ── 후처리 (mode별 추출 함수)
    if mode in ("opening", "role_reversal"):
        speech = _postprocess_speech(_extract_delimited_text(raw))
        is_valid = _is_valid_speech
    else:
        speech = _postprocess_speech(_extract_rebuttal_text(raw))
        is_valid = _is_valid_rebuttal

    # ── 무효 시 재시도 (이전 로직 복원)
    if not is_valid(speech):
        logger.warning("[write_node:%s] speech 무효, 재시도", mode)
        if mode in ("opening", "role_reversal"):
            messages.append(AIMessage(content=raw))
            messages.append(HumanMessage(content="한국어로만 작성하세요.\n\n### 답변 시작\n### 답변 끝"))
        else:
            messages.append(AIMessage(content=raw))
            messages.append(HumanMessage(content="반드시 한국어로만 3~4문장으로 반박하세요."))
        retry = invoke_with_retry(llm, messages, label=f"{mode}_retry")
        raw = retry.content if isinstance(retry.content, str) else str(retry.content)
        if mode in ("opening", "role_reversal"):
            speech = _postprocess_speech(_extract_delimited_text(raw))
        else:
            speech = _postprocess_speech(_extract_rebuttal_text(raw))

    # ── 영어 잔재 감지 → LLM 수정 요청 (이전 로직 복원)
    eng_words = re.findall(r'(?<![a-zA-Z])[a-z]{4,}(?![a-zA-Z])', speech)
    if eng_words:
        logger.warning("[write_node:%s] 영어 감지: %s → 수정 요청", mode, eng_words[:3])
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(
            content=f'다음 영어 단어를 한국어로 바꿔서 다시 작성하라: {", ".join(eng_words[:5])}\n한국어만 사용. 같은 형식 유지.'
        ))
        fix = invoke_with_retry(llm, messages, label=f"{mode}_fix_eng")
        raw_fix = fix.content if isinstance(fix.content, str) else str(fix.content)
        if mode in ("opening", "role_reversal"):
            fixed = _postprocess_speech(_extract_delimited_text(raw_fix))
        else:
            fixed = _postprocess_speech(_extract_rebuttal_text(raw_fix))
        if is_valid(fixed):
            speech = fixed
            raw = raw_fix

    # ── 최종 fallback
    if not is_valid(speech):
        logger.warning("[write_node:%s] fallback 사용", mode)
        if mode in ("opening", "role_reversal"):
            speech = f"### 자기소개와 입장 표명\n저는 {stance_kr} 입장입니다.\n\n### 결론\n{stance_kr} 입장을 유지합니다."
        elif mode == "synthesis":
            speech = "이 문제에 대해 구체적인 해결 방안을 함께 논의해야 합니다."
        else:
            speech = "상대의 주장은 핵심 전제가 부족합니다. 따라서 설득력이 없습니다."

    return {"speech": speech, "raw": raw}


# ── Review 노드 ───────────────────────────────────────────────────────────

def review_node(state: TurnState) -> dict:
    """Reviewer: 품질 검증."""
    speech = state.get("speech", "")
    expected_stance = state.get("expected_stance", "PRO")
    topic = state.get("topic", "")
    mode = state.get("mode", "attack")

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
