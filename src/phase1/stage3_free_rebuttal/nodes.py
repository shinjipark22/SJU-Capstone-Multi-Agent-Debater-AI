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
                from src.graph.llm import safe_search_invoke
                result = safe_search_invoke(tc.get("args", {}))
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
    my_previous: str = "",
) -> str:
    """상대 발언을 공격하는 프롬프트. 상대 입론도 참고하여 모순을 찾는다."""
    ref_block = ""
    if search_results:
        ref_block = f"\n[참고 자료 — 반박 근거로 활용하라]\n{search_results}\n"

    opp_block = ""
    if opp_opening:
        opp_block = f"\n[상대 입론 — 지금 발언과 모순되는 부분이 있으면 지적하라]\n{opp_opening[:300]}\n"

    prev_block = ""
    if my_previous:
        prev_block = (
            f"\n[너의 이전 라운드 발언 — 같은 사례·논거·표현 반복 금지, "
            f"다른 측면·새로운 사례로 공격하라]\n{my_previous[:500]}\n"
        )

    return f"""상대 발언:
{target_argument}
{ref_block}{opp_block}{prev_block}
반드시 아래 3단 구조로 반박하라:
1) 상대 핵심 주장을 1문장으로 요약
2) 그 주장의 가장 치명적 약점 1개를 지목
3) 그 약점만 집중 공격 (2~3문장)

[규칙]
- 반드시 합니다체
- 총 3~4문장 이내. 절대 5문장을 넘기지 마라
- 핵심 주장에 **강조** 표시
- 직전 2턴에서 사용한 핵심 논지를 반복하지 마라. 새로운 공격 축을 제시하라
- 구체적 수치·기관명을 쓰려면 참고 자료에 있는 것만 인용. 참고 자료에 없으면 수치 없이 논리로 공격하라. 머릿속 수치는 쓰지 마라

반드시 아래 형식으로만 출력:

### 반박 시작
(반박 내용)
### 반박 끝"""


# ── 방어 프롬프트 (사용자 공격에 대한 방어) ─────────────────────────────────

def _plan_independent_attack(
    agent: Dict,
    topic: str,
    stance: str,
    opp_opening: str = "",
    my_previous: str = "",
) -> Tuple[Dict, str]:
    """Round 2 전용 — 사용자 입론과 무관한 새 공격 각도 + 검색 쿼리 CoT.

    Round 1 은 사용자 입론에서 논거 추출해 공격 → round 2 도 같은 방식이면
    같은 측면 반복. Round 2 는 사용자가 입론에서 안 다룬 새 측면을 자기
    진영 옹호 방향으로 도출 + 그 측면 검색 쿼리.

    Returns:
        (plan_dict, raw_response)
        plan_dict = {"attack_angle": str, "search_query": str}
    """
    from src.phase1.stage1_opening.nodes import _invoke_with_retry, _llm, _extract_plan_json

    stance_kr = "찬성" if stance == "PRO" else "반대"
    plan_prompt = f"""[Round 2 공격 계획] '{topic}' 자유논박 두 번째 라운드의 공격 각도를 설계하라.

[사용자 입론 — Round 1 에서 이미 공격함. 이번엔 직접 다루지 마라]
{(opp_opening or "(없음)")[:500]}

[너의 Round 1 공격 — 같은 측면·사례·자료 반복 금지]
{(my_previous or "(없음)")[:400]}

[지시]
{stance_kr} 진영 옹호 입장에서, 사용자가 입론에서 다루지 않은 **새로운 측면** 으로 공격할 각도를 찾아라.
사용자 발언의 흠을 잡는 게 아니라, 자기 진영 옹호 논거를 새 각도로 던지는 것이다.

[출력 형식 — JSON 객체 하나만]
```json
{{
  "attack_angle": "새 공격 각도 1문장 ({stance_kr} 진영 옹호 방향, 사용자가 안 다룬 측면)",
  "search_query": "그 각도 뒷받침 자료 검색용 쿼리"
}}
```

[규칙]
- 사용자 입론의 측면 재사용 금지
- Round 1 공격과 완전히 다른 측면
- {stance_kr} 진영 옹호 방향만"""

    messages = [
        SystemMessage(content=agent["system_prompt"]),
        HumanMessage(content=plan_prompt),
    ]
    response = _invoke_with_retry(_llm, messages, label="round2_plan")
    raw = response.content if isinstance(response.content, str) else str(response.content)
    parsed = _extract_plan_json(raw)
    return parsed, raw


def _build_independent_attack_prompt(
    new_angle: str,
    search_results: str = "",
    opp_opening: str = "",
    my_previous: str = "",
) -> str:
    """Round 2 전용 attack prompt — 사용자 입론과 무관한 새 각도 공격."""
    ref_block = ""
    if search_results:
        ref_block = f"\n[참고 자료 — 새 공격 각도 뒷받침용]\n{search_results}\n"

    opp_block = ""
    if opp_opening:
        opp_block = (
            f"\n[사용자 입론 — Round 1 에서 이미 다뤘으므로 이번엔 직접 인용·반박하지 마라]\n"
            f"{opp_opening[:300]}\n"
        )

    prev_block = ""
    if my_previous:
        prev_block = (
            f"\n[너의 Round 1 공격 — 같은 사례·논거·자료 반복 금지]\n{my_previous[:500]}\n"
        )

    return f"""이번 라운드 (Round 2) 는 사용자 입론을 직접 공격하지 않고, 사용자가 아직 다루지 않은
**새로운 공격 각도** 를 제시하라.

[너의 새 공격 각도]
{new_angle}
{ref_block}{opp_block}{prev_block}
[작성 구조]
1) 새 공격 각도를 1문장으로 명확히 제시
2) 참고 자료의 사실·사례로 그 각도를 뒷받침 (자료 없으면 논리로)
3) 사용자가 이 각도에 어떻게 답할지 묻는 질문 1개

[규칙]
- 반드시 합니다체. 총 3~4문장 이내
- 핵심 주장에 **강조** 표시
- 사용자 입론을 직접 인용·반박하지 마라 — 새 측면을 던져라
- Round 1 공격과 다른 키워드·사례·자료 사용

반드시 아래 형식으로만 출력:

### 반박 시작
(반박 내용)
### 반박 끝"""


def _build_defense_prompt(
    user_attack: str,
    my_opening: str = "",
    search_results: str = "",
    my_previous: str = "",
) -> str:
    """사용자의 공격에 대해 나의 입론을 근거로 방어하는 프롬프트."""
    my_block = ""
    if my_opening:
        my_block = f"\n[나의 입론 — 이것을 근거로 방어하라]\n{my_opening[:300]}\n"

    ref_block = ""
    if search_results:
        ref_block = f"\n[참고 자료]\n{search_results}\n"

    prev_block = ""
    if my_previous:
        prev_block = (
            f"\n[너의 이전 라운드 발언 — 같은 사례·근거 반복 금지, "
            f"새로운 사례·다른 측면으로 반격하라]\n{my_previous[:500]}\n"
        )

    return f"""상대의 공격:
{user_attack}
{my_block}{ref_block}{prev_block}
반드시 아래 구조로 방어하라:
1) 상대가 질문했으면 그 질문에 1문장으로 직접 답하라
2) 상대 공격의 논리적 허점 1개를 지적하라
3) 새로운 근거 1개로 반격하라 (이전에 쓴 근거 반복 금지)

[규칙]
- 반드시 합니다체
- 총 3~4문장 이내. 절대 5문장을 넘기지 마라
- 핵심 주장에 **강조** 표시
- 구체적 수치·기관명을 쓰려면 참고 자료에 있는 것만 인용. 없으면 수치 없이 논리로 방어하라. 머릿속 수치는 쓰지 마라
- 직전 2턴에서 사용한 논지를 반복하지 마라

반드시 아래 형식으로만 출력:

### 반박 시작
(반박 내용 — 3~4문장)
### 반박 끝"""


# ── 공통 setup ─────────────────────────────────────────────────────────────

def _setup_free_rebuttal(state: DebateState) -> Dict:
    """defense / attack 노드가 공통으로 쓰는 셋업 — opponent, history, display 등."""
    selected_id = state.get("selected_opponent_id")
    if not selected_id:
        raise ValueError("[free_rebuttal] selected_opponent_id가 필요합니다.")

    agent_map = {a["agent_id"]: a for a in state["agents"]}
    if selected_id not in agent_map:
        raise ValueError(f"[free_rebuttal] 존재하지 않는 에이전트: {selected_id}")

    opponent = agent_map[selected_id]
    try:
        from src.graph.vector_store import set_current_stance
        set_current_stance(opponent.get("stance"))
    except Exception:
        pass

    speaking_order = state["speaking_order"]
    stance_nums = build_agent_stance_nums(state["agents"], speaking_order)
    opponent_display = (
        f"{'찬성' if opponent['stance'] == 'PRO' else '반대'}"
        f"{stance_nums.get(selected_id, 0)}"
    )
    return {
        "selected_id": selected_id,
        "opponent": opponent,
        "history": list(state["debate_history"]),
        "opponent_display": opponent_display,
    }


# ── 메인 노드 ──────────────────────────────────────────────────────────────

def free_rebuttal_defense_node(state: DebateState) -> DebateState:
    """방어 단계만 생성. 첫 턴이거나 사용자 공격이 없으면 entry 추가 없이 통과.

    노드 단위 분리 이유: SSE 가 발화 단위로 즉시 push 되도록.
    원래는 free_rebuttal_node 한 번에 방어+공격을 생성해 두 entry 가 동시에 history 에
    추가됐고, frontend 가 시간차 없이 두 발화를 한꺼번에 받았다.
    """
    _opening_mod._used_doc_ids = set()
    setup = _setup_free_rebuttal(state)
    history = setup["history"]
    selected_id = setup["selected_id"]
    opponent = setup["opponent"]
    current_turn = state["current_turn"]

    user_entries = [e for e in history if e["speaker_id"] == "user" and e["phase"] == "free_rebuttal"]
    agent_entries = [e for e in history if e["speaker_id"] == selected_id and e["phase"] == "free_rebuttal"]
    is_first_turn = len(agent_entries) == 0
    user_latest_attack = user_entries[-1]["content"] if user_entries else ""

    print(f"\n[3단계: 자유 논박] 사용자 ↔ {setup['opponent_display']}\n")

    # 첫 턴이거나 사용자 공격이 없으면 방어 스킵 (state 불변)
    if is_first_turn or not user_latest_attack:
        print(f"  [방어 스킵] 첫 턴 또는 사용자 공격 없음 — 공격 단계로 직진\n")
        return DebateState(**state)

    final_turn = is_final_agent_turn(state)
    print(f"  [Step 1 - 답변] 사용자 공격에 방어{' (최종 답변)' if final_turn else ''}\n")

    my_opening = next(
        (e["content"] for e in history if e["phase"] == "opening" and e["speaker_id"] == selected_id),
        "",
    )
    chain = _build_message_chain(opponent, history, selected_id, opponent["stance"])

    # 자기의 직전 자유논박 발언 — prompt 의 [이전 라운드 발언] 블록으로 박아
    # round 간 같은 사례·논거 반복을 명시적으로 차단한다.
    my_previous_in_phase = next(
        (e["content"] for e in reversed(history)
         if e.get("phase") == "free_rebuttal" and e.get("speaker_id") == selected_id),
        "",
    )

    # Pre-search: 사용자 공격 내용 기반 검색
    from src.phase1.stage2_rebuttal.nodes import _pre_search_rebuttal
    def_search_results, def_pre_tc = _pre_search_rebuttal(state["topic"], user_latest_attack)
    tool_calls_log: List[Dict] = list(def_pre_tc)

    defense_prompt = _build_defense_prompt(
        user_latest_attack, my_opening, search_results=def_search_results,
        my_previous=my_previous_in_phase,
    )
    defense, raw_def, tc_def = _generate_with_chain(list(chain), defense_prompt)
    tool_calls_log.extend(tc_def)

    history.append(DebateEntry(
        turn=current_turn,
        speaker_id=selected_id,
        stance=opponent["stance"],
        phase="free_rebuttal",
        content=defense,
        target_id="user",
        tool_calls_log=tool_calls_log,
        json_raw=raw_def,
    ))
    print(f"  [{setup['opponent_display']} - 답변] (turn={current_turn})")
    print(f"  {defense}\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn + 1,
        "phase": "free_rebuttal",
    })


def free_rebuttal_attack_node(state: DebateState) -> DebateState:
    """공격 단계만 생성. 마지막 턴이면 entry 추가 없이 통과.

    직전 방어가 history 에 들어가 있으면 _build_message_chain 이 자동으로 포함하므로
    별도로 chain 에 수동 append 할 필요 없음.
    """
    setup = _setup_free_rebuttal(state)
    history = setup["history"]
    selected_id = setup["selected_id"]
    opponent = setup["opponent"]
    current_turn = state["current_turn"]

    final_turn = is_final_agent_turn(state)
    if final_turn:
        print(f"  [마지막 턴] 공격 생략 — 방어만 수행\n")
        print(f"[3단계: 자유 논박] 에이전트 발언 완료 → 사용자 발언 대기\n")
        return DebateState(**{**state, "phase": "free_rebuttal"})

    user_entries = [e for e in history if e["speaker_id"] == "user" and e["phase"] == "free_rebuttal"]
    agent_entries = [e for e in history if e["speaker_id"] == selected_id and e["phase"] == "free_rebuttal"]

    user_latest_defense = (
        user_entries[-2]["content"] if len(user_entries) >= 2
        else (user_entries[-1]["content"] if user_entries else "")
    )
    opp_opening = next(
        (e["content"] for e in history if e["phase"] == "opening" and e["speaker_id"] == "user"),
        "",
    )

    if user_latest_defense:
        target_argument = user_latest_defense
    else:
        target_argument = _pick_one_argument(opp_opening)

    # Round 구분: free_rebuttal_user_turns counter 활용
    # round 1 attack 시점: counter = 0 (사용자 첫 attack 전)
    # round 2 attack 시점: counter = 1 (사용자 round 1 attack 후)
    is_round2 = state.get("free_rebuttal_user_turns", 0) >= 1

    # 이전 공격·약점 분석 (중복 방지)
    prev_attacks = [e["content"][:100] for e in agent_entries]
    prev_attacks_text = "\n".join(f"- {a}" for a in prev_attacks[-3:]) if prev_attacks else ""
    prev_weaknesses = "\n".join(
        log.get("result", "")
        for e in agent_entries
        for log in e.get("tool_calls_log", [])
        if log.get("name") == "analyze_weakness" and log.get("result")
    )
    # 자기 직전 자유논박 발언 — my_previous 블록으로 명시 전달.
    # prev_attacks_text 가 100자 단위 요약이라 부족 — 직전 full 발언을 따로 박는다.
    my_previous_full = next(
        (e["content"] for e in reversed(history)
         if e.get("phase") == "free_rebuttal" and e.get("speaker_id") == selected_id),
        "",
    )

    from src.phase1.stage2_rebuttal.nodes import _pre_search_rebuttal
    tool_calls_log: List[Dict] = []

    if is_round2:
        # ── Round 2: CoT 로 사용자 입론과 무관한 새 공격 각도 + 검색 ──────────
        print(f"  [Step 2 - Round 2 공격: CoT 로 새 각도 도출 후 공격]\n")
        plan, plan_raw = _plan_independent_attack(
            opponent, state["topic"], opponent["stance"],
            opp_opening=opp_opening, my_previous=my_previous_full,
        )
        new_angle = (plan.get("attack_angle") or "").strip() or _pick_one_argument(opp_opening)
        search_query = (plan.get("search_query") or "").strip() or new_angle
        tool_calls_log.append({
            "name": "round2_plan",
            "args": {"opp_opening_chars": len(opp_opening or "")},
            "result": (plan_raw or "")[:500],
        })

        from src.graph.llm import safe_search_invoke
        atk_result = safe_search_invoke({"query": search_query})
        atk_result = _truncate_tool_result(str(atk_result))
        tool_calls_log.append({
            "name": "search_web",
            "args": {"query": search_query},
            "result": atk_result,
        })
        print(f"  [Round 2 각도] {new_angle[:80]}")
        print(f"  [Round 2 쿼리] {search_query[:80]}")

        attack_prompt = _build_independent_attack_prompt(
            new_angle=new_angle,
            search_results=atk_result,
            opp_opening=opp_opening,
            my_previous=my_previous_full,
        )
    else:
        # ── Round 1: 기존 흐름 (사용자 입론 기반 공격) ──────────────────────
        print(f"  [Step 2 - Round 1 공격: 사용자 입론 허점 공격]\n")
        weakness = analyze_weakness(
            target_argument, state["topic"], prev_weaknesses=prev_weaknesses,
        )
        if weakness:
            tool_calls_log.append({"name": "analyze_weakness", "result": weakness})
            print(f"  [약점 분석] {weakness[:60]}\n")

        atk_search_results, atk_pre_tc = _pre_search_rebuttal(state["topic"], target_argument)
        tool_calls_log.extend(atk_pre_tc)

        weakness_hint = (
            f"\n[약점 분석 — 이 부분을 집중 공격하라]\n{weakness}\n"
            if weakness else ""
        )
        prev_hint = (
            f"\n[이전 공격 — 아래 내용은 이미 사용했으니 반복 금지. 완전히 다른 관점으로 공격하라]\n{prev_attacks_text}\n"
            if prev_attacks_text else ""
        )
        ref_block = atk_search_results if atk_search_results else ""
        attack_prompt = _build_attack_prompt(
            target_argument,
            search_results=(ref_block + weakness_hint + prev_hint),
            opp_opening=opp_opening,
            my_previous=my_previous_full,
        )
    # 직전 방어가 history 에 있으면 chain 에 자동 포함됨
    chain = _build_message_chain(opponent, history, selected_id, opponent["stance"])
    attack, raw_atk, tc_atk = _generate_with_chain(chain, attack_prompt)
    tool_calls_log.extend(tc_atk)

    attack_question = _generate_attack_question(attack, opponent["stance"], state["topic"])
    if attack_question:
        tool_calls_log.append({"name": "attack_question", "result": attack_question})
        print(f"  [공격 질문] {attack_question[:60]}\n")
    if "?" not in attack:
        q = attack_question if attack_question else "이에 대해 상대는 어떻게 설명하시겠습니까?"
        attack = attack.rstrip() + " " + q

    history.append(DebateEntry(
        turn=current_turn,
        speaker_id=selected_id,
        stance=opponent["stance"],
        phase="free_rebuttal",
        content=attack,
        target_id="user",
        tool_calls_log=tool_calls_log,
        json_raw=raw_atk,
    ))
    print(f"  [{setup['opponent_display']} - 공격] (turn={current_turn})")
    print(f"  {attack}\n")
    print(f"[3단계: 자유 논박] 에이전트 발언 완료 → 사용자 발언 대기\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn + 1,
        "phase": "free_rebuttal",
    })


def free_rebuttal_node(state: DebateState) -> DebateState:
    """방어 + 공격을 순차로 한 번에 처리. 노드 분리 전 호출 호환용."""
    state = free_rebuttal_defense_node(state)
    return free_rebuttal_attack_node(state)


# ── 턴 라우터 ──────────────────────────────────────────────────────────────

def should_end_free_rebuttal(state: DebateState) -> bool:
    """자유논박 종료 조건: 사용자 2턴 완료."""
    return state.get("free_rebuttal_user_turns", 0) >= 2


def is_final_agent_turn(state: DebateState) -> bool:
    """에이전트 최종 답변(공격 없음) 차례인지: 사용자 2턴 완료 상태."""
    return state.get("free_rebuttal_user_turns", 0) >= 2
