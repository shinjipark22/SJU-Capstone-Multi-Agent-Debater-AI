"""
nodes.py — 4단계: 역할 반전(Role Reversal) 노드

[동작 흐름]
    1. 상대팀 AI 에이전트 중 랜덤 1명을 대표로 선정
    2. 대표 AI가 자기 원래 입장의 반대(= 사용자 팀 입장)를 옹호하는 발언 생성
    3. 사용자도 상대팀 입장을 옹호하는 발언을 API로 제출 (노드 외부)

[설계 노트]
    - 입론과 동일한 구조 (논거 2개 + 결론), 자기소개 제외
    - 기존 토론 히스토리에서 상대팀 입론을 참고하여 역할반전 발언 작성
    - DeepSeek-R1-Distill-Qwen-14B 최적화
"""

from __future__ import annotations

import logging
import random
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import src.stage1_opening.nodes as _opening_mod
from src.stage1_opening.nodes import (
    _invoke_with_retry,
    _postprocess_speech,
    _extract_delimited_text,
    _is_valid_speech,
    _pre_search,
    _truncate_tool_result,
    search_web,
    _LLM_KWARGS,
)
from src.stage2_rebuttal.nodes import build_agent_stance_nums
from src.state import DebateEntry, DebateState

# ── 역할반전 전용 LLM ──────────────────────────────────────────────────────
_rr_llm = ChatOpenAI(**{**_LLM_KWARGS, "max_tokens": 4096, "temperature": 0.6})


# ── 상대팀 대표 랜덤 선정 ──────────────────────────────────────────────────

def _select_representative(
    agents: List[Dict],
    user_stance: str,
) -> Dict:
    """상대팀(사용자 반대편) AI 에이전트 중 랜덤 1명을 대표로 선정한다."""
    opposite_stance = "CON" if user_stance == "PRO" else "PRO"
    candidates = [a for a in agents if a["stance"] == opposite_stance]
    if not candidates:
        raise ValueError(f"[role_reversal] 상대팀({opposite_stance}) 에이전트가 없습니다.")
    return random.choice(candidates)


# ── 역할반전 프롬프트 ──────────────────────────────────────────────────────

def _build_role_reversal_prompt(
    topic: str,
    reversed_stance: str,
    agent_name: str,
    search_results: str,
    opponent_openings: str,
) -> str:
    """역할반전 발언 프롬프트를 생성한다.

    Args:
        topic            : 토론 주제
        reversed_stance  : 반전된 입장 (이 발언에서 옹호할 입장)
        agent_name       : 에이전트 표시명
        search_results   : 사전 검색 결과
        opponent_openings: 반전된 입장의 기존 입론들 (참고용)
    """
    stance_kr = "찬성" if reversed_stance == "PRO" else "반대"

    return f"""[역할 반전 단계]
너는 원래 {stance_kr}의 반대 입장이었지만, 지금은 역할을 반전하여 **{stance_kr} 입장을 옹호**해야 한다.
상대방의 관점에서 진심으로 설득력 있는 주장을 펼쳐라.

아래 참고 자료와 기존 {stance_kr}측 입론을 바탕으로 '{topic}'에 대한 {stance_kr} 입론을 작성하라.

[기존 {stance_kr}측 입론 — 참고하되 그대로 베끼지 말 것]
{opponent_openings}

[참고 자료]
{search_results}

[구조]
- 논거 2개, 각 3줄 이내
- 핵심 문장에 **강조** 사용

[인용 규칙]
- 참고 자료의 수치/기관명을 반드시 인용할 것
- 참고 자료에 없는 수치를 지어내지 마라
- 과장 금지. 데이터 범위 내에서만 주장
- 원문 그대로 인용. 자체 계산·환율 변환 금지
- 출처는 국제기구, 연구기관, 대학, 기업만 밝힐 것 (블로그·커뮤니티·개인 사이트 제외)

[형식]
- 한국어로 작성. 고유명사(기관명, 인명, 기술명)만 영어 허용. 그 외 모든 서술은 한국어로
- 합니다체(격식체)

반드시 아래 형식으로만 출력:

### 답변 시작
### 논거 1: 소제목
(논거)
### 논거 2: 소제목
(논거)
### 결론
(결론)
### 답변 끝"""


# ── 역할반전 발언 생성 ────────────────────────────────────────────────────

def _generate_role_reversal(agent: Dict, prompt: str) -> Tuple[str, str]:
    """역할반전 발언 생성. 입론과 동일한 패턴."""
    messages = [
        SystemMessage(content=agent["system_prompt"]),
        HumanMessage(content=prompt),
    ]

    response: AIMessage = _invoke_with_retry(_rr_llm, messages, label="role_reversal")
    raw = response.content if isinstance(response.content, str) else str(response.content)
    speech = _postprocess_speech(_extract_delimited_text(raw))

    if not _is_valid_speech(speech):
        logger.warning("[role_reversal] speech 무효, 재시도")
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content="한국어로만 역할반전 발언을 작성하세요.\n\n### 답변 시작\n(발언)\n### 답변 끝"))
        retry: AIMessage = _invoke_with_retry(_rr_llm, messages, label="role_reversal_retry")
        raw = retry.content if isinstance(retry.content, str) else str(retry.content)
        speech = _postprocess_speech(_extract_delimited_text(raw))

    # 영어 잔재 감지 → 수정 요청
    eng_words = re.findall(r'(?<![a-zA-Z])[a-z]{4,}(?![a-zA-Z])', speech)
    if eng_words:
        logger.warning("[role_reversal] 영어 감지: %s → 수정 요청", eng_words[:3])
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content=f'다음 영어 단어를 한국어로 바꿔서 다시 작성하라: {", ".join(eng_words[:5])}\n한국어만 사용. 같은 형식 유지.'))
        fix: AIMessage = _invoke_with_retry(_rr_llm, messages, label="role_reversal_fix_eng")
        raw_fix = fix.content if isinstance(fix.content, str) else str(fix.content)
        fixed = _postprocess_speech(_extract_delimited_text(raw_fix))
        if _is_valid_speech(fixed):
            speech = fixed
            raw = raw_fix

    return speech, raw


# ── 메인 노드 ──────────────────────────────────────────────────────────────

def role_reversal_node(state: DebateState) -> DebateState:
    """4단계 역할반전 노드.

    상대팀 대표 AI 1명이 사용자 팀 입장을 옹호하는 발언을 생성한다.
    사용자는 별도 API로 상대팀 입장 옹호 발언을 제출한다.
    """
    _opening_mod._used_doc_ids = set()

    topic = state["topic"]
    user_stance = state["user_stance"]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]
    speaking_order = state["speaking_order"]
    stance_nums = build_agent_stance_nums(state["agents"], speaking_order)

    # ── 상대팀 대표 선정 (랜덤)
    representative = _select_representative(state["agents"], user_stance)
    rep_id = representative["agent_id"]
    original_stance = representative["stance"]
    reversed_stance = "PRO" if original_stance == "CON" else "CON"  # 반전된 입장 = 사용자 팀 입장

    rep_label = "찬성" if original_stance == "PRO" else "반대"
    rep_display = f"{rep_label} 에이전트{stance_nums.get(rep_id, 1)}"
    reversed_label = "찬성" if reversed_stance == "PRO" else "반대"

    print(f"\n[4단계: 역할 반전]")
    print(f"  상대팀 대표: {rep_display} (원래: {rep_label} → 반전: {reversed_label})")
    print(f"  사용자: (원래: {'찬성' if user_stance == 'PRO' else '반대'} → 반전: {'반대' if user_stance == 'PRO' else '찬성'})\n")

    # ── 반전된 입장의 기존 입론 수집 (참고용)
    opponent_openings = []
    for entry in history:
        if entry["phase"] == "opening" and entry["stance"] == reversed_stance:
            opponent_openings.append(entry["content"][:300])
    openings_text = "\n---\n".join(opponent_openings) if opponent_openings else "(없음)"

    # ── 사전 검색 (반전된 입장의 근거)
    print(f"  [{rep_display}] 역할반전 발언 생성 중...")
    search_results, tool_calls_log = _pre_search(
        topic, reversed_stance,
        topic_id=state.get("topic_id", ""),
    )

    # ── 프롬프트 구성 + LLM 호출
    prompt = _build_role_reversal_prompt(
        topic=topic,
        reversed_stance=reversed_stance,
        agent_name=rep_display,
        search_results=search_results,
        opponent_openings=openings_text,
    )
    final_text, raw = _generate_role_reversal(representative, prompt)

    # ── fallback
    if not _is_valid_speech(final_text):
        logger.warning("[role_reversal] fallback 사용")
        final_text = (
            f"### 논거 1: 관점의 전환\n"
            f"역할을 반전하여 생각해보면, {reversed_label} 입장에도 타당한 근거가 있습니다.\n\n"
            f"### 논거 2: 균형 잡힌 시각\n"
            f"양측의 주장을 모두 고려할 때, {reversed_label} 관점의 장점도 인정해야 합니다.\n\n"
            f"### 결론\n"
            f"{reversed_label} 입장에서 바라보면, 이 주제에 대해 충분한 설득력이 있습니다."
        )

    # ── 발언 기록
    entry = DebateEntry(
        turn=current_turn,
        speaker_id=rep_id,
        stance=reversed_stance,  # 반전된 입장으로 기록
        phase="role_reversal",
        content=final_text,
        target_id=None,
        tool_calls_log=tool_calls_log,
        json_raw=raw,
    )
    history.append(entry)
    current_turn += 1

    print(f"  [{rep_display}] 역할반전 발언 완료 (turn={entry['turn']})")
    print(f"  {final_text[:100]}...\n")
    print(f"[4단계: 역할 반전] AI 발언 완료 → 사용자 역할반전 발언 대기\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "phase": "role_reversal",
        "role_reversed": True,
    })
