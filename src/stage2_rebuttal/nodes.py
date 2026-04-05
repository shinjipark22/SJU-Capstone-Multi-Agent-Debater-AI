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

# ── 연쇄논박 전용 LLM (max_tokens=2048: think 블록 + 답변 여유) ──────────────
_rebuttal_llm = ChatOpenAI(**{**_LLM_KWARGS, "max_tokens": 2048})


# ── delimiter 추출 (연쇄논박 전용, 엄격) ─────────────────────────────────────

def _extract_rebuttal_text(content: str) -> str:
    """### 답변 시작 ~ ### 답변 끝 사이만 추출. 없으면 첫 문단만 반환."""
    text = content.strip()

    # <think> 제거
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = text.replace('</think>', '').strip()

    # 영어 줄 제거 (한글이 없고 영어가 포함된 줄은 모두 제거)
    lines = text.split('\n')
    text = '\n'.join(l for l in lines if not l.strip() or re.search(r'[가-힣]', l)).strip()

    # delimiter 추출
    m = re.search(r'###\s*답변\s*시작\s*(?:###)?\s*\n?(.*?)\n?\s*###\s*답변\s*끝', text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()

    m = re.search(r'###\s*답변\s*시작\s*(?:###)?\s*\n?(.*)', text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()

    # fallback: 한국어 문장 2~4개 추출
    logger.warning("[_extract_rebuttal_text] delimiter 없음, 한국어 문장 추출")
    korean_lines = [l.strip() for l in text.split('\n') if l.strip() and re.search(r'[가-힣]', l)]
    if korean_lines:
        return '\n'.join(korean_lines[:4])
    return text.strip()


# ── 반박 프롬프트 ────────────────────────────────────────────────────────────

def _pre_search_rebuttal(topic: str, target_speech: str, stance: str, focus_area: str) -> Tuple[str, List[Dict]]:
    """연쇄논박용 사전검색. 상대 발언 키워드 + focus_area 기반."""
    tool_calls_log: List[Dict] = []
    results = []

    # 상대 발언에서 핵심 키워드 추출 (첫 50자)
    snippet = target_speech[:50].replace("\n", " ")
    focus_hint = focus_area.replace("검색 방향: ", "").strip() if focus_area else topic

    query = f"{topic} {focus_hint}"
    tool_calls_log.append({"name": "search_web", "args": {"query": query}})
    web_result = search_web.invoke({"query": query})
    results.append(_truncate_tool_result(web_result))

    return "\n".join(results), tool_calls_log


def _build_rebuttal_prompt(
    target_speech: str,
    target_display: str,
    stance_kr: str,
    my_previous: str = "",
    focus_area: str = "",
    search_results: str = "",
) -> str:
    prev_block = ""
    if my_previous:
        prev_block = f"\n[내가 이전에 한 발언 — 같은 내용 반복 금지]\n{my_previous}\n"

    focus_block = ""
    if focus_area:
        focus_hint = focus_area.replace("검색 방향: ", "").strip()
        focus_block = f"\n[공격 관점]\n다음 관점에서 상대를 공격하라: {focus_hint}\n"

    search_block = ""
    if search_results:
        search_block = f"\n[참고 자료 — 자연스럽게 활용]\n{search_results}\n"

    return f"""상대 발언:
{target_speech}
{prev_block}{focus_block}{search_block}
상대 주장의 핵심 논리를 무너뜨려라.

구조:
- 첫 문장: 상대 주장의 오류를 지적 (매번 다른 표현 사용)
- 둘째 문장: 왜 틀렸는지 설명
- 셋째 문장: 근거 또는 사례
- 마지막 문장: 결론 ({stance_kr} 입장 강화)

조건:
- 3~4문장만 작성
- 상대 주장 1개만 공격
- 한국어만 사용
- 반드시 "~입니다/~습니다" 존댓말 사용
- 공격적으로, 짧고 명확하게
- 이전 발언과 다른 논점을 공격하라

금지:
- 5문장 이상
- 번호 매김 (1. 2. 3. 등)
- 설명형/중립 표현
- 입론 내용 반복
- 존재하지 않는 데이터 생성

근거 규칙:
- 참고 자료의 내용을 자연스럽게 녹여서 반박
- 참고 자료에 없는 수치/통계는 사용 금지
- 검색 도구 이름을 언급하지 마라

반드시 아래 형식으로만 출력:

### 답변 시작
여기에 반박만 작성
### 답변 끝"""


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

    # delimiter 둘 다 없으면 1회 재시도
    if '### 답변 시작' not in raw or '### 답변 끝' not in raw:
        logger.warning("[rebuttal] delimiter 누락, 재시도")
        messages.append(HumanMessage(
            content='출력 형식이 틀렸다. ### 답변 시작 ### 와 ### 답변 끝 ### 사이에 한국어 반박만 작성하라.'
        ))
        retry: AIMessage = _invoke_with_retry(_rebuttal_llm, messages, label="rebuttal_retry")
        raw = retry.content if isinstance(retry.content, str) else str(retry.content)
        speech = _postprocess_speech(_extract_rebuttal_text(raw))

    # fallback: 최후의 수단 (None 또는 5자 미만)
    if speech is None or len(speech.strip()) < 5:
        logger.warning("[rebuttal] fallback 사용")
        stance_kr = "찬성" if stance == "PRO" else "반대"
        speech = (
            f"{target_display}의 주장은 핵심 전제가 부족하다. "
            f"따라서 설득력이 없다. "
            f"나는 {stance_kr} 입장을 유지한다."
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
        focus_area=focus,
        search_results=search_results,
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

        if not pair["awaiting_response"]:
            if attacker_id == "user":
                print(f"  [라운드 {round_num}] 공격: 사용자 → {target_id} (API 대기)\n")
                continue

            agent = agent_map[attacker_id]
            slabel = "찬성" if agent["stance"] == "PRO" else "반대"
            display = f"{slabel} 에이전트{stance_nums[attacker_id]}"
            print(f"  [라운드 {round_num}] 공격: {display} → {target_id}")

            entry = generate_ai_rebuttal(
                topic=topic, history=history, agent=agent,
                target_id=target_id, stance_num=stance_nums[attacker_id],
                target_stance_num=stance_nums.get(target_id, 0),
                current_turn=current_turn, is_response=False,
            )
            history.append(entry)
            current_turn += 1
            pairs[pair_idx]["awaiting_response"] = True
            print(f"  [라운드 {round_num}] 공격 완료 (turn={entry['turn']})\n")

        if pairs[pair_idx]["awaiting_response"] and not pairs[pair_idx]["done"]:
            if target_id == "user":
                print(f"  [라운드 {round_num}] 응답: 사용자 ← {attacker_id} (API 대기)\n")
                continue

            agent = agent_map[target_id]
            slabel = "찬성" if agent["stance"] == "PRO" else "반대"
            display = f"{slabel} 에이전트{stance_nums[target_id]}"
            print(f"  [라운드 {round_num}] 응답: {display} ← {attacker_id}")

            entry = generate_ai_rebuttal(
                topic=topic, history=history, agent=agent,
                target_id=attacker_id, stance_num=stance_nums[target_id],
                target_stance_num=stance_nums.get(attacker_id, 0),
                current_turn=current_turn, is_response=True,
            )
            history.append(entry)
            current_turn += 1
            pairs[pair_idx]["done"] = True
            print(f"  [라운드 {round_num}] 응답 완료 (turn={entry['turn']})\n")

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
