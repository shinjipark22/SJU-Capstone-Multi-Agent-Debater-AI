"""
nodes.py — 3단계: 자유 논박(Free Rebuttal) 노드

[동작 흐름]
    연쇄논박 스타일의 라운드 반복:
    1. 에이전트 → [공격] 사용자 입론 허점 공격 (단발 생성)
    2. 사용자  → [답변] + [공격] (API에서 2개 입력)
    3. 에이전트 → [답변] 사용자 공격에 방어 + [공격] 사용자 입론 다른 허점 공격
    4. 반복

[설계 노트]
    - 매 턴이 연쇄논박과 동일한 단발 생성 (멀티턴 챗봇 아님)
    - Qwen2.5-1.5B 조건부 검색
    - DeepSeek-R1-Distill-Qwen-14B 최적화
"""

from __future__ import annotations

import logging
import random
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import src.phase1.stage1_opening.nodes as _opening_mod
from src.phase1.stage1_opening.nodes import (
    _invoke_with_retry,
    _postprocess_speech,
    _truncate_tool_result,
    search_web,
    _LLM_KWARGS,
)
from langchain_core.messages import ToolMessage as _ToolMessage
from src.phase1.stage2_rebuttal.nodes import (
    _extract_rebuttal_text,
    _is_valid_rebuttal,
    build_agent_stance_nums,
    analyze_weakness,
    _analysis_llm,
)
from src.state import DebateEntry, DebateState

# ── 자유논박 전용 LLM ───────────────────────────────────────────────────────
_fr_llm = ChatOpenAI(**{**_LLM_KWARGS, "max_tokens": 512, "temperature": 0.6})
_fr_llm_with_tools = _fr_llm.bind_tools([search_web])


# ── 공격 질문 생성 (자유논박 전용) ─────────────────────────────────────────

def _generate_attack_question(attack_text: str, stance: str, topic: str) -> str:
    """공격 발언의 흐름에 맞는 마무리 질문을 생성한다."""
    try:
        messages = [
            HumanMessage(content=f"""다음 공격 발언을 읽고, 상대를 곤란하게 만드는 함정형 질문 1개를 만들어라.

공격 발언: {attack_text[:300]}

규칙:
- 양자택일형 또는 기준 확인형 질문. 상대가 어떻게 답해도 불리해지는 구조
- "어떻게 생각하십니까?" 같은 열린 질문 금지
- 한국어, 합니다체, 한 문장만, ?로 끝나야 함
- 설명 없이 질문만 출력

좋은 예시:
- "귀측은 원인을 '위험 증가'로 정의합니까, '실제 이주 계기'로 정의합니까?"
- "동일 재해에서 국가별 이주 규모가 다른 이유를 기후만으로 설명할 수 있습니까?"
- "이 사례를 일반 규칙으로 확대할 근거가 있습니까?"

나쁜 예시:
- "어떻게 생각하십니까?"
- "인정하지 않으시겠습니까?"
""")
        ]
        response = _analysis_llm.invoke(messages)
        result = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
        for line in result.split('\n'):
            line = line.strip()
            if line.endswith('?') and re.search(r'[가-힣]', line):
                return line
        return ""
    except Exception as e:
        logger.warning("[32B] _generate_attack_question 오류: %s", e)
        return ""


# ── 입론에서 논거 추출 (연쇄논박과 동일) ────────────────────────────────────



def _pick_one_argument(speech: str) -> str:
    """입론에서 논거 1 또는 논거 2를 랜덤으로 하나만 추출한다."""
    parts = re.split(r'###\s*논거\s*\d+\s*[:：]?', speech)
    arguments = []
    for i, p in enumerate(parts):
        if i == 0:
            continue
        conclusion_idx = p.find('### 결론')
        if conclusion_idx != -1:
            p = p[:conclusion_idx]
        text = p.strip()
        if text and len(text) > 20:
            arguments.append(text)
    if arguments:
        return random.choice(arguments)
    return speech


# ── 멀티턴 메시지 체인 구축 ───────────────────────────────────────────────

def _build_message_chain(
    agent: Dict,
    history: List,
    selected_id: str,
    stance: str,
) -> List:
    """전체 토론 히스토리를 멀티턴 메시지 체인으로 구축한다.

    1~3단계 전체 발언을 포함하여 이전 맥락을 참조 가능.
    """
    from src.graph.llm import build_debate_chain
    from src.phase0.persona_factory import INTENSITY_PROFILES

    stance_kr = "찬성" if stance == "PRO" else "반대"
    intensity = agent.get("intensity", 3)
    intensity_style = INTENSITY_PROFILES.get(intensity, INTENSITY_PROFILES[3])["style"]

    system = (
        f"{agent['system_prompt']}\n\n"
        f"[최우선 규칙] 너는 {stance_kr} 입장이다. "
        f"3~4문장. 한국어. 합니다체.\n"
        f"[논증 스타일] {intensity_style}"
    )
    messages = [SystemMessage(content=system)]

    # 전체 토론 히스토리를 메시지 체인으로 (입론~자유논박 모두 포함)
    debate_chain = build_debate_chain(history, selected_id)
    messages.extend(debate_chain)

    return messages


def _generate_with_chain(
    messages: List,
    prompt: str,
) -> Tuple[str, str, List[Dict]]:
    """멀티턴 체인에 새 프롬프트를 추가하고 tool calling으로 생성한다."""
    messages.append(HumanMessage(content=prompt))
    tc_log: List[Dict] = []

    response: AIMessage = _invoke_with_retry(_fr_llm_with_tools, messages, label="free_rebuttal")

    # tool call 처리
    if hasattr(response, 'tool_calls') and response.tool_calls:
        messages.append(response)
        for tc in response.tool_calls:
            if tc.get("name") == "search_web":
                result = search_web.invoke(tc.get("args", {}))
                result = _truncate_tool_result(str(result))
                tc_log.append({"name": "search_web", "args": tc.get("args", {}), "result": result})
                messages.append(_ToolMessage(content=result, tool_call_id=tc.get("id", "")))
                logger.info("[free_rebuttal] tool call: search_web(%s)", tc.get("args"))
        response = _invoke_with_retry(_fr_llm, messages, label="free_rebuttal_with_search")

    raw = response.content if isinstance(response.content, str) else str(response.content)
    speech = _postprocess_speech(_extract_rebuttal_text(raw))

    # 무효 시 1회 재시도
    if not _is_valid_rebuttal(speech):
        logger.warning("[free_rebuttal] speech 무효 → 재시도")
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content="반드시 한국어로만 3~4문장으로 반박하세요."))
        retry: AIMessage = _invoke_with_retry(_fr_llm, messages, label="free_rebuttal_retry")
        raw = retry.content if isinstance(retry.content, str) else str(retry.content)
        speech = _postprocess_speech(_extract_rebuttal_text(raw))

    # fallback
    if not _is_valid_rebuttal(speech):
        logger.warning("[free_rebuttal] fallback 사용")
        speech = "상대의 주장은 핵심 전제가 부족합니다. 따라서 설득력이 없습니다."

    return speech, raw, tc_log


# ── 공격 프롬프트 (상대 입론 논거 공격) ─────────────────────────────────────

def _build_attack_prompt(
    target_argument: str,
    search_results: str = "",
    opp_opening: str = "",
) -> str:
    """상대 발언을 공격하는 프롬프트. 상대 입론도 참고하여 모순을 찾는다."""
    ref_block = ""
    if search_results:
        ref_block = f"\n[참고 자료 — 반박 근거로 활용하라]\n{search_results}\n"

    opp_block = ""
    if opp_opening:
        opp_block = f"\n[상대 입론 — 지금 발언과 모순되는 부분이 있으면 지적하라]\n{opp_opening[:300]}\n"

    return f"""상대 발언:
{target_argument}
{ref_block}{opp_block}
반드시 아래 3단 구조로 반박하라:
1) 상대 핵심 주장을 1문장으로 요약
2) 그 주장의 가장 치명적 약점 1개를 지목
3) 그 약점만 집중 공격 (2~3문장)

[규칙]
- 반드시 합니다체
- 총 3~4문장 이내. 절대 5문장을 넘기지 마라
- 핵심 주장에 **강조** 표시
- 직전 2턴에서 사용한 핵심 논지를 반복하지 마라. 새로운 공격 축을 제시하라
- 수치를 지어내지 마라

반드시 아래 형식으로만 출력:

### 반박 시작
(반박 내용)
### 반박 끝"""


# ── 방어 프롬프트 (사용자 공격에 대한 방어) ─────────────────────────────────

def _build_defense_prompt(
    user_attack: str,
    my_opening: str = "",
    search_results: str = "",
) -> str:
    """사용자의 공격에 대해 나의 입론을 근거로 방어하는 프롬프트."""
    my_block = ""
    if my_opening:
        my_block = f"\n[나의 입론 — 이것을 근거로 방어하라]\n{my_opening[:300]}\n"

    ref_block = ""
    if search_results:
        ref_block = f"\n[참고 자료]\n{search_results}\n"

    return f"""상대의 공격:
{user_attack}
{my_block}{ref_block}
반드시 아래 구조로 방어하라:
1) 상대가 질문했으면 그 질문에 1문장으로 직접 답하라
2) 상대 공격의 논리적 허점 1개를 지적하라
3) 새로운 근거 1개로 반격하라 (이전에 쓴 근거 반복 금지)

[규칙]
- 반드시 합니다체
- 총 3~4문장 이내. 절대 5문장을 넘기지 마라
- 핵심 주장에 **강조** 표시
- 수치를 지어내지 마라
- 직전 2턴에서 사용한 논지를 반복하지 마라

반드시 아래 형식으로만 출력:

### 반박 시작
(반박 내용 — 3~4문장)
### 반박 끝"""


# ── 메인 노드 ──────────────────────────────────────────────────────────────

def free_rebuttal_node(state: DebateState) -> DebateState:
    """3단계 자유 논박 노드 (멀티턴 메시지 체인).

    이전 자유논박 대화 히스토리를 메시지 체인으로 구축하여
    LLM이 대화 흐름을 기억한 상태에서 답변/공격을 생성한다.
    """
    _opening_mod._used_doc_ids = set()

    # ── 상대 에이전트 확인
    selected_id = state.get("selected_opponent_id")
    if not selected_id:
        raise ValueError("[free_rebuttal] selected_opponent_id가 필요합니다.")

    agent_map = {a["agent_id"]: a for a in state["agents"]}
    if selected_id not in agent_map:
        raise ValueError(f"[free_rebuttal] 존재하지 않는 에이전트: {selected_id}")

    opponent = agent_map[selected_id]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]
    speaking_order = state["speaking_order"]
    stance_nums = build_agent_stance_nums(state["agents"], speaking_order)

    opponent_display = (
        f"{'찬성' if opponent['stance'] == 'PRO' else '반대'}"
        f"{stance_nums.get(selected_id, 0)}"
    )
    print(f"\n[3단계: 자유 논박] 사용자 ↔ {opponent_display}\n")

    # ── 입론 추출
    my_opening = ""
    opp_opening = ""
    for e in history:
        if e["phase"] == "opening" and e["speaker_id"] == selected_id:
            my_opening = e["content"]
        if e["phase"] == "opening" and e["speaker_id"] == "user":
            opp_opening = e["content"]

    # ── 멀티턴 메시지 체인 구축
    chain = _build_message_chain(opponent, history, selected_id, opponent["stance"])

    user_entries = [e for e in history if e["speaker_id"] == "user" and e["phase"] == "free_rebuttal"]
    agent_entries = [e for e in history if e["speaker_id"] == selected_id and e["phase"] == "free_rebuttal"]

    speeches = []
    tool_calls_log: List[Dict] = []
    is_first_turn = len(agent_entries) == 0
    final_turn = is_final_agent_turn(state)  # 마지막 턴이면 방어만, 공격 없음

    # 사용자 최근 발언 분리 (답변 + 공격이 별개)
    user_latest_defense = user_entries[-2]["content"] if len(user_entries) >= 2 else (user_entries[-1]["content"] if user_entries else "")
    user_latest_attack = user_entries[-1]["content"] if user_entries else ""

    # 이전 공격 내용 + 약점 분석 결과 수집 (중복 방지용)
    prev_attacks = [e["content"][:100] for e in agent_entries]
    prev_attacks_text = "\n".join(f"- {a}" for a in prev_attacks[-3:]) if prev_attacks else ""
    prev_weaknesses = "\n".join(
        log.get("result", "")
        for e in agent_entries
        for log in e.get("tool_calls_log", [])
        if log.get("name") == "analyze_weakness" and log.get("result")
    )

    # ── Step 1: 답변 (사용자의 공격에 대한 방어)
    if not is_first_turn and user_latest_attack:
        print(f"  [Step 1 - 답변] 사용자 공격에 방어{' (최종 답변)' if final_turn else ''}\n")

        defense_prompt = _build_defense_prompt(user_latest_attack, my_opening)
        defense, raw_def, tc_def = _generate_with_chain(list(chain), defense_prompt)
        tool_calls_log.extend(tc_def)
        speeches.append(("답변", defense, raw_def))

    # ── Step 2: 공격 (마지막 턴이면 스킵 — 사용자 응답 기회 없으므로)
    if not final_turn:
        if not is_first_turn and user_latest_defense:
            target_argument = user_latest_defense
        else:
            target_argument = _pick_one_argument(opp_opening)

        print(f"  [Step 2 - 공격] 상대 논거 허점 공격\n")

        # 약점 분석 (이전 분석과 다른 약점 요청)
        weakness = analyze_weakness(target_argument, state["topic"], prev_weaknesses=prev_weaknesses)
        if weakness:
            tool_calls_log.append({"name": "analyze_weakness", "result": weakness})
            print(f"  [약점 분석] {weakness[:60]}\n")

        weakness_hint = f"\n[약점 분석 — 이 부분을 집중 공격하라]\n{weakness}\n" if weakness else ""
        prev_hint = f"\n[이전 공격 — 아래 내용은 이미 사용했으니 반복 금지. 완전히 다른 관점으로 공격하라]\n{prev_attacks_text}\n" if prev_attacks_text else ""
        attack_prompt = _build_attack_prompt(target_argument, weakness_hint + prev_hint, opp_opening)
        # 답변이 있으면 그 결과를 체인에 추가한 뒤 공격
        attack_chain = list(chain)
        if speeches:
            attack_chain.append(AIMessage(content=speeches[0][1]))  # 답변을 체인에 포함
        attack, raw_atk, tc_atk = _generate_with_chain(attack_chain, attack_prompt)
        tool_calls_log.extend(tc_atk)

        # 공격 결과를 읽고 맥락에 맞는 질문 생성
        attack_question = _generate_attack_question(attack, opponent["stance"], state["topic"])
        if attack_question:
            tool_calls_log.append({"name": "attack_question", "result": attack_question})
            print(f"  [공격 질문] {attack_question[:60]}\n")

        # 후처리: 공격 끝에 질문 추가
        if "?" not in attack:
            q = attack_question if attack_question else "이에 대해 상대는 어떻게 설명하시겠습니까?"
            attack = attack.rstrip() + " " + q

        speeches.append(("공격", attack, raw_atk))
    else:
        print(f"  [마지막 턴] 공격 생략 — 방어만 수행\n")

    # ── 발언 기록
    for label, speech, raw in speeches:
        history.append(DebateEntry(
            turn=current_turn,
            speaker_id=selected_id,
            stance=opponent["stance"],
            phase="free_rebuttal",
            content=speech,
            target_id="user",
            tool_calls_log=tool_calls_log,
            json_raw=raw,
        ))
        current_turn += 1

        print(f"  [{opponent_display} - {label}] (turn={current_turn - 1})")
        print(f"  {speech}\n")

    print(f"[3단계: 자유 논박] 에이전트 발언 완료 → 사용자 발언 대기\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "phase": "free_rebuttal",
    })


# ── 턴 라우터 ──────────────────────────────────────────────────────────────

def should_end_free_rebuttal(state: DebateState) -> bool:
    """자유논박 종료 조건: 사용자 2턴 완료."""
    return state.get("free_rebuttal_user_turns", 0) >= 2


def is_final_agent_turn(state: DebateState) -> bool:
    """에이전트 최종 답변(공격 없음) 차례인지: 사용자 2턴 완료 상태."""
    return state.get("free_rebuttal_user_turns", 0) >= 2
