"""
nodes.py — 2단계: 연쇄 논박(Chained Rebuttal) 노드

[설계 노트]
    - delimiter 기반 자연어 출력
    - 단일 LLM 호출 (max_tokens=256)
    - DeepSeek-R1-Distill-Qwen-14B 최적화
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import src.stage1_opening.nodes as _opening_mod
from src.stage1_opening.nodes import (
    _invoke_with_retry,
    _postprocess_speech,
    _truncate_tool_result,
    search_web,
    search_vector_db,
    _LLM_KWARGS,
)
from src.state import (
    DebateEntry,
    DebateState,
    build_chained_rebuttal_pairs,
)

# ── 연쇄논박 전용 LLM (max_tokens=1024: think 제한 + 간결한 답변 유도) ────────
_rebuttal_llm = ChatOpenAI(**{**_LLM_KWARGS, "max_tokens": 1024})


# ── 텍스트 추출 (delimiter 없이, <think> + 영어 제거 후 한국어만) ────────────

def _extract_rebuttal_text(content: str) -> str:
    """<think> 블록과 영어를 제거하고 한국어 문장만 추출한다."""
    text = content.strip()

    # <think> 제거
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = text.replace('</think>', '').strip()

    # 영어 CoT 제거: 문장별로 한글 비율이 30% 미만이면 삭제
    parts = re.split(r'(?<=[.!?])\s+', text)
    cleaned = []
    for p in parts:
        if not p.strip():
            continue
        korean_chars = len(re.findall(r'[가-힣]', p))
        total_alpha = len(re.findall(r'[a-zA-Z가-힣]', p))
        if total_alpha > 0 and korean_chars / total_alpha < 0.3:
            continue  # 영어 비중 70% 이상 → CoT로 판단
        cleaned.append(p)
    text = ' '.join(cleaned)

    # 한국어가 포함된 줄만 추출
    korean_lines = []
    for l in text.split('\n'):
        s = l.strip()
        if not s or not re.search(r'[가-힣]', s):
            continue
        # 메타 문장 제거
        if s.startswith('상대의 주장을 반박') or s.startswith('반박'):
            if len(s) < 15:
                continue
        # 번호 매김 제거 (줄 시작 + 문장 중간)
        s = re.sub(r'^\d+\.\s*', '', s)
        s = re.sub(r'\s+\d+\.\s+', ' ', s)
        s = re.sub(r'^3\.\s*1\.\s*', '', s)  # "3. 1." 패턴
        korean_lines.append(s)
    return '\n'.join(korean_lines) if korean_lines else text.strip()


# ── 반박 프롬프트 ────────────────────────────────────────────────────────────

def _pre_search_rebuttal(topic: str, target_speech: str, stance: str, focus_area: str) -> Tuple[str, List[Dict]]:
    """연쇄논박용 사전검색. 토픽 + focus_area 기반."""
    tool_calls_log: List[Dict] = []
    results = []

    focus_hint = focus_area.replace("검색 방향: ", "").strip() if focus_area else topic
    query = f"{topic} {focus_hint}"
    tool_calls_log.append({"name": "search_web", "args": {"query": query}})
    web_result = search_web.invoke({"query": query})
    results.append(_truncate_tool_result(web_result))

    return "\n".join(results), tool_calls_log


def _extract_key_claim(speech: str) -> str:
    """상대 발언에서 핵심 주장 1문장을 추출한다."""
    # 결론 섹션 우선
    m = re.search(r'(?:결론|따라서|그러므로)[^\n]*', speech)
    if m:
        return m.group().strip()
    # 마지막 한국어 문장
    sentences = [s.strip() for s in speech.replace('\n', ' ').split('.') if s.strip() and re.search(r'[가-힣]', s)]
    if sentences:
        return sentences[-1] + '.'
    return speech[:100]


def _build_rebuttal_prompt(
    target_speech: str,
    target_display: str,
    stance_kr: str,
    my_previous: str = "",
    search_results: str = "",
    attack_style: str = "",
) -> str:
    context = ""
    if search_results:
        context += f"\n[참고 자료]\n{search_results}\n"
    if my_previous:
        context += f"\n[이전 발언 — 같은 내용 반복 금지]\n{my_previous}\n"

    return f"""너는 {stance_kr} 입장이다. 너의 역할은 "분석자"가 아니라 "공격자"다.

[상대 발언]
{target_speech}
{context}
[공격 방식]
{attack_style}

규칙:
- 상대 주장을 평가하거나 분석하지 마라
- 설명하지 마라
- 중립적 표현 금지
- 오직 상대 주장의 오류를 공격하는 문장만 작성하라

3~4문장. ~입니다/~습니다 체.
번호 매김(1. 2. 3.) 절대 금지. 목록 금지.
자연스러운 문단으로 이어서 작성하라."""


# ── 반박 생성 ────────────────────────────────────────────────────────────────

def _generate_rebuttal_speech(
    agent: Dict,
    prompt: str,
    target_display: str,
    stance: str,
) -> Tuple[str, str]:
    """단일 LLM 호출(max_tokens=256). delimiter 없으면 1회 재시도."""
    messages = [
        SystemMessage(content=agent["system_prompt"]),
        HumanMessage(content=prompt),
    ]

    response: AIMessage = _invoke_with_retry(_rebuttal_llm, messages, label="rebuttal")
    raw = response.content if isinstance(response.content, str) else str(response.content)
    speech = _postprocess_speech(_extract_rebuttal_text(raw))

    # fallback: None 또는 5자 미만
    if speech is None or len(speech.strip()) < 5:
        logger.warning("[rebuttal] fallback 사용")
        stance_kr = "찬성" if stance == "PRO" else "반대"
        speech = (
            f"{target_display}의 주장은 핵심 전제가 부족합니다. "
            f"따라서 설득력이 없습니다. "
            f"저는 {stance_kr} 입장을 유지합니다."
        )

    return speech, raw


# ── 공개 유틸리티 ────────────────────────────────────────────────────────────

def build_agent_stance_nums(
    agents: List[Dict],
    speaking_order: List[str],
) -> Dict[str, int]:
    agent_map = {a["agent_id"]: a for a in agents}
    counter: Dict[str, int] = {"PRO": 0, "CON": 0}
    nums: Dict[str, int] = {}
    for sid in speaking_order:
        if sid == "user" or sid not in agent_map:
            continue
        counter[agent_map[sid]["stance"]] += 1
        nums[sid] = counter[agent_map[sid]["stance"]]
    return nums


_ATTACK_STYLES = [
    "전제 공격: 상대 주장에 깔린 가정이 틀렸음을 지적하라",
    "현실성 공격: 실제 상황에서 작동하지 않는다는 점을 지적하라",
    "부작용 공격: 해당 주장으로 인해 발생하는 문제를 강조하라",
    "비교 공격: 더 나은 대안이 있음을 제시하라",
    "데이터 공격: 상대 근거의 신뢰성이나 부족함을 지적하라",
]

# 에이전트별 공격 방식 카운터 (같은 방식 반복 방지)
_attack_counter: Dict[str, int] = {}


def generate_ai_rebuttal(
    topic: str,
    history: List[DebateEntry],
    agent: Dict,
    target_id: str,
    stance_num: int,
    target_stance_num: int,
    current_turn: int,
    is_response: bool,
) -> DebateEntry:
    target_speech = "(발언 기록 없음)"
    target_stance = "CON" if agent["stance"] == "PRO" else "PRO"
    for entry in reversed(history):
        if entry["speaker_id"] == target_id:
            target_speech = entry["content"]
            target_stance = entry["stance"]
            break

    # 자신의 이전 발언 추출 (반복 방지)
    my_previous = ""
    for entry in reversed(history):
        if entry["speaker_id"] == agent["agent_id"] and entry["phase"] == "chained_rebuttal":
            my_previous = entry["content"][:200]
            break

    # 공격 방식 순환 할당
    aid = agent["agent_id"]
    idx = _attack_counter.get(aid, 0)
    attack_style = _ATTACK_STYLES[idx % len(_ATTACK_STYLES)]
    _attack_counter[aid] = idx + 1

    t_label = "찬성" if target_stance == "PRO" else "반대"
    target_display = f"{t_label} 에이전트{target_stance_num}" if target_id != "user" else "사용자"
    stance_kr = "찬성" if agent["stance"] == "PRO" else "반대"

    focus = agent.get("focus_area", "")

    # 사전검색
    search_results, search_log = _pre_search_rebuttal(
        topic=topic, target_speech=target_speech,
        stance=agent["stance"], focus_area=focus,
    )

    prompt = _build_rebuttal_prompt(
        target_speech=target_speech,
        target_display=target_display,
        stance_kr=stance_kr,
        my_previous=my_previous,
        search_results=search_results,
        attack_style=attack_style,
    )

    speech, raw = _generate_rebuttal_speech(
        agent=agent, prompt=prompt,
        target_display=target_display, stance=agent["stance"],
    )

    return DebateEntry(
        turn=current_turn, speaker_id=agent["agent_id"],
        stance=agent["stance"], phase="chained_rebuttal",
        content=speech, target_id=target_id,
        tool_calls_log=search_log, json_raw=raw,
    )


# ── 메인 노드 ─────────────────────────────────────────────────────────────────

def chained_rebuttal_node(state: DebateState) -> DebateState:
    _opening_mod._used_doc_ids = set()

    topic = state["topic"]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]
    agent_map = {a["agent_id"]: a for a in state["agents"]}

    pairs: List[Dict] = [dict(p) for p in (state["rebuttal_pairs"] or [])]
    if not pairs:
        pairs = [dict(p) for p in build_chained_rebuttal_pairs(
            state["agents"], state["user_stance"],
        )]

    stance_nums = build_agent_stance_nums(state["agents"], state["speaking_order"])

    print(f"\n[2단계: 연쇄 논박] 총 {len(pairs)}개 라운드\n")

    for pair_idx, pair in enumerate(pairs):
        if pair["done"]:
            continue

        round_num = pair["round"]
        attacker_id = pair["attacker_id"]
        target_id = pair["target_id"]

        # 공격만 수행 (응답 턴 제거)
        if attacker_id == "user":
            print(f"  [라운드 {round_num}] 공격: 사용자 → {target_id} (API 대기)\n")
            continue

        agent = agent_map[attacker_id]
        slabel = "찬성" if agent["stance"] == "PRO" else "반대"
        display = f"{slabel} 에이전트{stance_nums[attacker_id]}"
        print(f"  [라운드 {round_num}] {display} → {target_id}")

        entry = generate_ai_rebuttal(
            topic=topic, history=history, agent=agent,
            target_id=target_id, stance_num=stance_nums[attacker_id],
            target_stance_num=stance_nums.get(target_id, 0),
            current_turn=current_turn, is_response=False,
        )
        history.append(entry)
        current_turn += 1
        pairs[pair_idx]["done"] = True
        print(f"  [라운드 {round_num}] 완료 (turn={entry['turn']})\n")

    all_done = all(p["done"] for p in pairs)
    next_phase = "free_rebuttal" if all_done else "chained_rebuttal"

    if all_done:
        print("[2단계: 연쇄 논박] 완료 → 3단계 자유 논박으로 전환\n")
    else:
        print("[2단계: 연쇄 논박] AI 논박 완료 → 사용자 논박 대기\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "rebuttal_pairs": pairs,
        "phase": next_phase,
    })
