"""
nodes.py — 2단계: 연쇄 논박(Chained Rebuttal) 노드

[동작 흐름]
    1. rebuttal_pairs가 없으면 build_chained_rebuttal_pairs()로 생성
    2. 각 미완료 라운드마다:
        a. 공격 발언 (awaiting_response=False):
           - 공격자가 타겟의 이전 발언을 분석하고 반박
        b. 응답 발언 (awaiting_response=True):
           - 타겟이 공격에 대해 방어 + 재반박
    3. 사용자(user) 차례는 건너뛰고 API에서 처리
    4. 모든 라운드 완료 후 phase를 "free_rebuttal"로 전환

[설계 노트]
    - 입론 단계의 도구(search_web, search_vector_db)와 LLM 인스턴스를 재사용한다.
    - 도구 실행 루프는 반박용 최종 프롬프트만 변경하여 동일 패턴을 따른다.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

# ── 입론 단계에서 정의한 도구·LLM·유틸리티 재사용 ─────────────────────────────────
import src.stage1_opening.nodes as _opening_mod
from src.stage1_opening.nodes import (
    _llm_json,
    _llm_with_tools,
    _parse_xml_tool_calls,
    _postprocess_speech,
    _TOOL_MAP,
)
from src.state import (
    DebateEntry,
    DebateState,
    RebuttalPair,
    build_chained_rebuttal_pairs,
)


# ── 응답 후처리 유틸리티 ───────────────────────────────────────────────────────

def _extract_rebuttal_from_json(content: str) -> Tuple[str, str, str]:
    """JSON 응답에서 speech와 target_agent를 추출한다.

    입론 단계의 _extract_speech_from_json과 동일한 3단계 폴백을 적용하되,
    target_agent 필드를 추가로 추출한다.

    Returns:
        (speech, target_agent, json_raw)
    """
    json_raw = content.strip()
    text = json_raw

    # 1. <think> 블록 제거 (JSON 모드에서도 CoT가 나올 수 있음)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = text.replace('</think>', '').strip()

    # 2. JSON 파싱 → speech + target_agent 추출
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "speech" in data:
            speech = data["speech"]
            target = data.get("target_agent", "")
            logger.info("[CoT reasoning] %s", data.get("reasoning", "")[:100])
            return _postprocess_speech(speech), target, json_raw
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. 폴백: JSON 블록 추출 재시도
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, dict) and "speech" in data:
                target = data.get("target_agent", "")
                return _postprocess_speech(data["speech"]), target, json_raw
        except (json.JSONDecodeError, TypeError):
            pass

    # 4. 최종 폴백: <tool_call> 제거 후 원본 반환
    logger.warning("[_extract_rebuttal_from_json] JSON 파싱 실패, 원본 텍스트 반환")
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    return _postprocess_speech(text.strip()), "", json_raw


# ── 반박용 도구 실행 루프 ────────────────────────────────────────────────────────

def _run_rebuttal_loop(messages: List, target_id: str) -> Tuple[str, str, List[Dict]]:
    """반박용 도구 실행 루프.

    입론 단계의 _run_tool_calling_loop와 동일한 패턴으로 도구를 반복 호출한 뒤,
    반박용 JSON({"reasoning", "target_agent", "speech"})을 생성한다.

    Args:
        messages:  [SystemMessage, HumanMessage, ...] 초기 메시지 리스트
        target_id: 반박 대상 에이전트 ID (JSON 출력 형식에 포함)

    Returns:
        (speech, json_raw, tool_calls_log)
    """
    tool_calls_log: List[Dict] = []

    while True:
        response: AIMessage = _llm_with_tools.invoke(messages)
        content: str = response.content if isinstance(response.content, str) else str(response.content)

        # ── 도구 호출 감지: 정식 파싱 우선, 없으면 XML 폴백 ──────────────────
        tool_calls = list(response.tool_calls) if response.tool_calls else _parse_xml_tool_calls(content)

        if not tool_calls:
            # 더 이상 도구 호출 없음 → JSON 모드로 최종 반박 생성
            messages.append(response)
            messages.append(HumanMessage(
                content='위 검색 결과를 바탕으로 연쇄 논박을 작성하세요. '
                        '반드시 {"reasoning": "분석 과정", '
                        f'"target_agent": "{target_id}", '
                        '"speech": "최종 발언"} JSON으로만 출력하세요.'
            ))
            json_response: AIMessage = _llm_json.invoke(messages)
            json_content: str = json_response.content if isinstance(json_response.content, str) else str(json_response.content)
            speech, _, json_raw = _extract_rebuttal_from_json(json_content)
            return speech, json_raw, tool_calls_log

        # ── 도구 호출 로그 수집 ───────────────────────────────────────────────
        for tc in tool_calls:
            tool_calls_log.append({"name": tc["name"], "args": tc["args"]})

        # ── 도구 실행 및 결과 주입 ────────────────────────────────────────────
        messages.append(response)

        if response.tool_calls:
            # 정식 파싱된 경우: ToolMessage로 1:1 주입
            for tc in tool_calls:
                if tc["name"] not in _TOOL_MAP:
                    tool_result: str = f"[오류] 알 수 없는 도구: {tc['name']}"
                else:
                    try:
                        tool_result = _TOOL_MAP[tc["name"]].invoke(tc["args"])
                    except (ValidationError, Exception) as e:
                        tool_result = f"[도구 호출 오류] {tc['name']} 인자가 잘못되었습니다: {e}"
                messages.append(ToolMessage(content=tool_result, tool_call_id=tc["id"]))
        else:
            # XML 폴백: tool_call_id 없으므로 HumanMessage로 결과 일괄 주입
            results = []
            for tc in tool_calls:
                if tc["name"] not in _TOOL_MAP:
                    tool_result = f"[오류] 알 수 없는 도구: {tc['name']}"
                else:
                    try:
                        tool_result = _TOOL_MAP[tc["name"]].invoke(tc["args"])
                    except (ValidationError, Exception) as e:
                        tool_result = f"[도구 호출 오류] {tc['name']} 인자가 잘못되었습니다: {e}"
                results.append(f"[{tc['name']} 결과]\n{tool_result}")
            messages.append(HumanMessage(
                content="\n\n".join(results) + "\n\n위 검색 결과를 바탕으로 연쇄 논박을 완성하세요."
            ))


# ── 내부 유틸리티 ─────────────────────────────────────────────────────────────

def _find_target_speech(
    history: List[DebateEntry],
    target_id: str,
) -> Optional[DebateEntry]:
    """debate_history에서 타겟의 가장 최근 발언을 찾는다."""
    for entry in reversed(history):
        if entry["speaker_id"] == target_id:
            return entry
    return None


def _build_rebuttal_prompt(
    topic: str,
    stance: str,
    target_id: str,
    target_speech: str,
    target_stance: str,
    stance_num: int,
    is_response: bool,
) -> str:
    """반박 요청 HumanMessage 본문을 생성한다."""
    stance_kr = "찬성(PRO)" if stance == "PRO" else "반대(CON)"
    stance_label = "찬성" if stance == "PRO" else "반대"
    agent_name = f"{stance_label} 에이전트{stance_num}"
    target_stance_kr = "찬성(PRO)" if target_stance == "PRO" else "반대(CON)"

    action = "방어 및 재반박" if is_response else "공격 반박"

    return f"""search_web과 search_vector_db를 호출해 근거를 수집한 뒤, '{topic}'에 대해 {target_id}의 발언을 {action}하는 연쇄 논박을 작성하세요.

[상대방 발언 — {target_id} ({target_stance_kr})]
{target_speech}

[출력 형식]
반드시 아래 JSON 형태로만 응답하세요:
{{"reasoning": "상대 발언 분석, 반박 전략 구상 (이 부분은 관중에게 보이지 않습니다)", "target_agent": "{target_id}", "speech": "아래 구조를 따르는 최종 토론 발언"}}

speech는 마크다운 형식으로 작성하세요. ## 소제목 뒤에는 반드시 줄바꿈 후 본문을 작성하세요.
**강조 표시**는 핵심적인 문장에 사용하되, 남용하지 마세요.

speech의 구조:
## 상대방 주장 요약
{target_id}의 핵심 주장과 논거를 1~2문장으로 요약한다.
## 모순 및 허점 지적
상대 논리의 모순점, 비약, 근거 부족을 구체적으로 지적한다.
## 근거 기반 반박
검색 결과를 바탕으로 상대 주장을 논파한다.
## 내 주장 강화
{stance_kr} 진영의 입장을 재강화하며 마무리한다.

[진영 고수 규칙]
1. 스탠스 고정: 무조건 {stance_kr} 입장만 방어하라. 상대 진영 논리에 동조하거나 타협하는 것은 절대 금지.
2. 불리한 정보 반박: 검색 결과에 당신의 진영에 불리한 내용이 있다면 절대 수용하지 마라. 반드시 "일각에서는 ~라 우려하지만" 형태의 예상 반론으로 삼아 철저히 논파하라.
3. 결론 일관성: 모든 발언의 마지막은 반드시 {stance_kr} 입장을 강력히 재확인하며 끝내라.

[주의] 한자(漢字), 일본어, 아랍 문자 등 외국 문자 사용 금지. 반드시 한글로만 작성하세요."""


# ── 공개 유틸리티 (노드·API 공용) ────────────────────────────────────────────

def build_agent_stance_nums(
    agents: List[Dict],
    speaking_order: List[str],
) -> Dict[str, int]:
    """에이전트별 진영 내 번호를 계산한다. (찬성 에이전트1, 반대 에이전트2 등)"""
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
    current_turn: int,
    is_response: bool,
) -> DebateEntry:
    """단일 AI 에이전트의 반박 발언을 생성한다.

    노드 내부 루프와 API 핸들러(사용자 공격 후 AI 응답 생성) 양쪽에서 사용한다.

    Args:
        topic:        토론 주제
        history:      현재까지의 debate_history (타겟 발언 조회용)
        agent:        발언할 AI 에이전트 스냅샷 (AgentSnapshot dict)
        target_id:    반박 대상 speaker_id
        stance_num:   진영 내 번호 (표시용)
        current_turn: 할당할 turn 번호
        is_response:  True면 방어/재반박, False면 공격 반박

    Returns:
        생성된 DebateEntry
    """
    target_entry = _find_target_speech(history, target_id)
    target_speech = target_entry["content"] if target_entry else "(발언 기록 없음)"
    target_stance = target_entry["stance"] if target_entry else (
        "CON" if agent["stance"] == "PRO" else "PRO"
    )

    messages = [
        SystemMessage(content=agent["system_prompt"]),
        HumanMessage(content=_build_rebuttal_prompt(
            topic, agent["stance"], target_id, target_speech,
            target_stance, stance_num, is_response,
        )),
    ]

    final_text, json_raw, tool_calls_log = _run_rebuttal_loop(messages, target_id)

    return DebateEntry(
        turn=current_turn,
        speaker_id=agent["agent_id"],
        stance=agent["stance"],
        phase="chained_rebuttal",
        content=final_text,
        target_id=target_id,
        tool_calls_log=tool_calls_log,
        json_raw=json_raw,
    )


# ── 메인 노드 ─────────────────────────────────────────────────────────────────

def chained_rebuttal_node(state: DebateState) -> DebateState:
    """2단계 연쇄 논박 노드.

    rebuttal_pairs의 각 라운드를 순서대로 처리한다.
    각 라운드는 (공격 → 응답) 2개 서브턴으로 구성되며,
    사용자(user) 차례는 건너뛰고 API를 통해 별도 처리한다.

    Args:
        state: 현재 DebateState (phase == "chained_rebuttal" 을 전제)

    Returns:
        debate_history가 누적되고, 모든 라운드 완료 시
        phase가 "free_rebuttal"(3단계 자유 논박)로 변경된 DebateState
    """
    # 문서 중복 추적 초기화 (새로운 단계이므로 리셋)
    _opening_mod._used_doc_ids = set()

    topic: str = state["topic"]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]
    agent_map = {a["agent_id"]: a for a in state["agents"]}

    # rebuttal_pairs 생성 (최초 진입 시)
    pairs: List[Dict] = [dict(p) for p in (state["rebuttal_pairs"] or [])]
    if not pairs:
        pairs = [dict(p) for p in build_chained_rebuttal_pairs(
            state["agents"], state["user_stance"],
        )]

    # 에이전트별 진영 번호 매핑
    stance_nums = build_agent_stance_nums(state["agents"], state["speaking_order"])

    print(f"\n[2단계: 연쇄 논박] 총 {len(pairs)}개 라운드\n")

    for pair_idx, pair in enumerate(pairs):
        if pair["done"]:
            continue

        round_num = pair["round"]
        attacker_id = pair["attacker_id"]
        target_id = pair["target_id"]

        # ── 공격 발언 ────────────────────────────────────────────────
        if not pair["awaiting_response"]:
            if attacker_id == "user":
                print(f"  [라운드 {round_num}] 공격: 사용자 → {target_id} (API 대기)\n")
                continue  # 사용자 턴은 API에서 처리

            agent = agent_map[attacker_id]
            stance_label = "찬성" if agent["stance"] == "PRO" else "반대"
            display_name = f"{stance_label} 에이전트{stance_nums[attacker_id]}"

            print(f"  [라운드 {round_num}] 공격: {display_name} → {target_id}")

            entry = generate_ai_rebuttal(
                topic=topic,
                history=history,
                agent=agent,
                target_id=target_id,
                stance_num=stance_nums[attacker_id],
                current_turn=current_turn,
                is_response=False,
            )
            history.append(entry)
            current_turn += 1
            pairs[pair_idx]["awaiting_response"] = True

            print(f"  [라운드 {round_num}] 공격 완료 (turn={entry['turn']})\n")

        # ── 응답 발언 ────────────────────────────────────────────────
        if pairs[pair_idx]["awaiting_response"] and not pairs[pair_idx]["done"]:
            if target_id == "user":
                print(f"  [라운드 {round_num}] 응답: 사용자 ← {attacker_id} (API 대기)\n")
                continue  # 사용자 턴은 API에서 처리

            agent = agent_map[target_id]
            stance_label = "찬성" if agent["stance"] == "PRO" else "반대"
            display_name = f"{stance_label} 에이전트{stance_nums[target_id]}"

            print(f"  [라운드 {round_num}] 응답: {display_name} ← {attacker_id}")

            entry = generate_ai_rebuttal(
                topic=topic,
                history=history,
                agent=agent,
                target_id=attacker_id,
                stance_num=stance_nums[target_id],
                current_turn=current_turn,
                is_response=True,
            )
            history.append(entry)
            current_turn += 1
            pairs[pair_idx]["done"] = True

            print(f"  [라운드 {round_num}] 응답 완료 (turn={entry['turn']})\n")

    # 완료 판단
    all_done = all(p["done"] for p in pairs)

    if all_done:
        next_phase = "free_rebuttal"
        print("[2단계: 연쇄 논박] 완료 → 3단계 자유 논박(free_rebuttal)으로 전환\n")
    else:
        next_phase = "chained_rebuttal"
        print("[2단계: 연쇄 논박] AI 논박 완료 → 사용자 논박 대기\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "rebuttal_pairs": pairs,
        "phase": next_phase,
    })
