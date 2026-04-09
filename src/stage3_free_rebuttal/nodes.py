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

import src.stage1_opening.nodes as _opening_mod
from src.stage1_opening.nodes import (
    _invoke_with_retry,
    _postprocess_speech,
    _truncate_tool_result,
    _has_cot_leakage,
    search_web,
    _LLM_KWARGS,
)
from src.stage2_rebuttal.nodes import (
    _extract_rebuttal_text,
    _decide_search,
    _is_valid_rebuttal,
    build_agent_stance_nums,
)
from src.state import DebateEntry, DebateState

# ── 자유논박 전용 LLM ───────────────────────────────────────────────────────
_fr_llm = ChatOpenAI(**{**_LLM_KWARGS, "max_tokens": 1024, "temperature": 0.6})


# ── 입론에서 논거 추출 (연쇄논박과 동일) ────────────────────────────────────

_ATTACK_STYLES = [
    "전제 공격: 상대 주장에 깔린 가정이 틀렸음을 지적하라",
    "현실성 공격: 실제 상황에서 작동하지 않는다는 점을 지적하라",
    "부작용 공격: 해당 주장으로 인해 발생하는 문제를 강조하라",
    "비교 공격: 더 나은 대안이 있음을 제시하라",
    "데이터 공격: 상대 근거의 신뢰성이나 부족함을 지적하라",
]
_attack_idx: int = 0


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


# ── 단발 생성 (연쇄논박 방식) ──────────────────────────────────────────────

def _generate_single_shot(
    agent: Dict,
    prompt: str,
    stance: str,
    topic: str,
) -> Tuple[str, str]:
    """연쇄논박과 동일한 단발 생성. CoT 유출 시 재시도."""
    stance_kr = "찬성" if stance == "PRO" else "반대"
    opposite_kr = "반대" if stance == "PRO" else "찬성"

    system = (
        f"{agent['system_prompt']}\n\n"
        f"[최우선 규칙] 너는 {stance_kr} 입장이다. "
        f"반드시 3~4문장으로만 답변하라. "
        f"상대 주장의 오류만 공격하라."
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ]

    response: AIMessage = _invoke_with_retry(_fr_llm, messages, label="free_rebuttal")
    raw = response.content if isinstance(response.content, str) else str(response.content)
    speech = _postprocess_speech(_extract_rebuttal_text(raw))

    # 무효 시 최대 3회 재시도 (1차: 대화 유지, 2차+: 대화 리셋)
    for retry_idx in range(3):
        if _is_valid_rebuttal(speech):
            break
        logger.warning("[free_rebuttal] speech 무효 → 재시도 %d/3", retry_idx + 1)
        if retry_idx == 0:
            messages.append(HumanMessage(content="반드시 한국어로만 3~4문장으로 반박하세요."))
        else:
            messages = [messages[0], HumanMessage(content=f"{prompt}\n\n반드시 한국어로만 답하라. 3~4문장.")]
        retry: AIMessage = _invoke_with_retry(_fr_llm, messages, label=f"free_rebuttal_retry{retry_idx}")
        raw = retry.content if isinstance(retry.content, str) else str(retry.content)
        speech = _postprocess_speech(_extract_rebuttal_text(raw))

    # 영어 잔재 감지 → LLM 수정 요청
    eng_words = re.findall(r'(?<![a-zA-Z])[a-z]{4,}(?![a-zA-Z])', speech)
    if eng_words:
        logger.warning("[free_rebuttal] 영어 감지: %s → 수정 요청", eng_words[:3])
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content=f'다음 영어 단어를 한국어로 바꿔서 다시 작성하라: {", ".join(eng_words[:5])}\n한국어만 사용. 같은 형식 유지.'))
        fix: AIMessage = _invoke_with_retry(_fr_llm, messages, label="free_rebuttal_fix_eng")
        raw_fix = fix.content if isinstance(fix.content, str) else str(fix.content)
        fixed = _postprocess_speech(_extract_rebuttal_text(raw_fix))
        if _is_valid_rebuttal(fixed):
            speech = fixed
            raw = raw_fix

    # fallback
    if not _is_valid_rebuttal(speech):
        logger.warning("[free_rebuttal] fallback 사용")
        speech = f"상대의 주장은 핵심 전제가 부족합니다. 따라서 설득력이 없습니다."

    return speech, raw


# ── 공격 프롬프트 (상대 입론 논거 공격) ─────────────────────────────────────

def _build_attack_prompt(
    target_argument: str,
    search_results: str = "",
) -> str:
    """상대 입론의 특정 논거를 공격하는 프롬프트."""
    ref_block = ""
    if search_results:
        ref_block = f"\n[참고 자료 — 반박 근거로 활용하라]\n{search_results}\n"

    global _attack_idx
    style = _ATTACK_STYLES[_attack_idx % len(_ATTACK_STYLES)]
    _attack_idx += 1

    return f"""상대 발언:
{target_argument}
{ref_block}
상대 발언에서 논리적 허점, 근거 부족, 과장된 주장을 찾아 공격하라. ({style})
- 상대가 인용한 수치/출처의 신뢰성을 검증하라
- 참고 자료에 반박 근거가 있으면 활용하라

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
상대가 너의 주장의 허점을 지적하고 있다.
- 상대가 질문했으면 그 질문에 먼저 직접 답하라
- 나의 입론이나 참고 자료에서 근거를 찾아 구체적으로 반박하라

[규칙]
- 3~4문장으로만 답변
- 핵심에 **강조** 사용
- 합니다체(격식체)
- 한국어로 작성. 고유명사만 영어 허용

반드시 아래 형식으로만 출력:

### 반박 시작
(반박 내용)
### 반박 끝"""


# ── 메인 노드 ──────────────────────────────────────────────────────────────

def free_rebuttal_node(state: DebateState) -> DebateState:
    """3단계 자유 논박 노드.

    연쇄논박 스타일 라운드 반복.
    매 호출마다 에이전트가 [답변] + [공격] 또는 [공격]만 생성.
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

    # ── 사용자의 최근 발언 찾기
    user_entries = [e for e in history if e["speaker_id"] == "user" and e["phase"] == "free_rebuttal"]
    agent_entries = [e for e in history if e["speaker_id"] == selected_id and e["phase"] == "free_rebuttal"]

    speeches = []
    tool_calls_log: List[Dict] = []
    user_latest = user_entries[-1]["content"] if user_entries else ""
    is_first_turn = len(agent_entries) == 0

    # ── Step 1: 답변 (상대 직전 턴에 대한 반박)
    if not is_first_turn and user_latest:
        print(f"  [Step 1 - 답변] 상대 직전 턴에 반박\n")

        query_def = _decide_search(user_latest, "")
        search_def = ""
        if query_def:
            tool_calls_log.append({"name": "search_web", "args": {"query": query_def}})
            web_result = search_web.invoke({"query": query_def})
            search_def = _truncate_tool_result(web_result)
            print(f"  [검색] '{query_def}'\n")

        defense_prompt = _build_defense_prompt(user_latest, my_opening, search_def)
        defense, raw_def = _generate_single_shot(opponent, defense_prompt, opponent["stance"], state["topic"])
        speeches.append(("답변", defense, raw_def))

    # ── Step 2: 공격 (직전 발언의 허점 공격, 첫 턴만 입론 공격)
    if user_latest and not is_first_turn:
        target_argument = user_latest  # 직전 발언의 허점 공격
    else:
        target_argument = _pick_one_argument(opp_opening)  # 첫 턴만 입론 공격

    print(f"  [Step 2 - 공격] 상대 논거 허점 공격\n")

    query_atk = _decide_search(target_argument, "")
    search_atk = ""
    if query_atk:
        tool_calls_log.append({"name": "search_web", "args": {"query": query_atk}})
        web_result = search_web.invoke({"query": query_atk})
        search_atk = _truncate_tool_result(web_result)
        print(f"  [검색] '{query_atk}'\n")

    attack_prompt = _build_attack_prompt(target_argument, search_atk)
    attack, raw_atk = _generate_single_shot(opponent, attack_prompt, opponent["stance"], state["topic"])
    speeches.append(("공격", attack, raw_atk))

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
