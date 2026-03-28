"""
nodes.py — Phase 1: 입론(Opening Arguments) 노드

[동작 흐름]
    1. speaking_order에서 AI 에이전트(user 제외)만 순서대로 추출
    2. 각 에이전트마다:
        a. system_prompt → SystemMessage, 입론 요청 → HumanMessage 구성
        b. LLM + 도구 실행 루프: tool_calls가 없어질 때까지 ToolMessage 주입 반복
        c. 최종 텍스트(입론)를 DebateEntry 형태로 debate_history에 누적
    3. 모든 AI 입론 완료 후 phase를 "chained_rebuttal"로 전환

[설계 노트]
    - state["phase"] 필드는 DebatePhase Literal 타입이므로 "chained_rebuttal" 사용
      (사용자가 "Phase 2: 자유 토론"이라 표현한 것은 2단계 연쇄 논박 단계를 의미함)
    - 도구 실행 루프는 각 에이전트 호출마다 독립적으로 동작 (메시지 격리)
"""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.state import DebateEntry, DebateState


# ── 도구 정의 ─────────────────────────────────────────────────────────────────

@tool
def search_web(query: str) -> str:
    """웹에서 최신 뉴스 및 정보를 검색합니다.

    Args:
        query: 검색할 키워드 또는 질문
    """
    # TODO: 실제 웹 검색 API(예: Tavily, SerpAPI) 연동
    return "[웹 검색 결과] 최신 뉴스 요약입니다."


@tool
def search_vector_db(query: str, topic: str, stance: str) -> str:
    """통합 VectorDB에서 진영·주제 필터링을 적용하여 전문가 문서를 검색합니다.

    Args:
        query:  검색할 내용 (자연어 질의)
        topic:  현재 토론 주제 (메타데이터 필터)
        stance: 검색할 진영 "PRO" 또는 "CON" (메타데이터 필터)
    """
    # TODO: vector_db.py의 실제 VectorDB 로직으로 교체
    return "[문서 검색 결과] 해당 진영과 주제에 맞는 전문가 문서입니다."


# ── 도구 이름 → 함수 매핑 ──────────────────────────────────────────────────────

_TOOLS: List = [search_web, search_vector_db]
_TOOL_MAP: Dict[str, Any] = {t.name: t for t in _TOOLS}

# ── LLM 초기화 (모듈 로드 시 1회) ─────────────────────────────────────────────
# vLLM OpenAI-compatible 서버를 사용한다.
# 서버 실행: vllm serve Qwen/Qwen3.5-9B --port 8000

_llm = ChatOpenAI(
    model="Qwen/Qwen3.5-9B",
    base_url="http://localhost:8000/v1",
    api_key="fake",          # vLLM은 API 키 불필요, 빈값 아닌 임의값 필요
    temperature=0.7,
)
_llm_with_tools = _llm.bind_tools(_TOOLS)


# ── 내부 유틸리티 ─────────────────────────────────────────────────────────────

def _build_opening_prompt(topic: str, stance: str) -> str:
    """입론 요청 HumanMessage 본문을 생성한다."""
    stance_kr = "찬성(PRO)" if stance == "PRO" else "반대(CON)"
    return (
        f"토론 주제: {topic}\n\n"
        f"당신의 진영은 {stance_kr}입니다.\n"
        f"위 주제에 대해 {stance_kr} 입장에서 논리적이고 설득력 있는 입론(Opening Argument)을 작성하세요.\n"
        f"필요하다면 search_web 또는 search_vector_db 도구를 사용하여 근거를 보강하세요.\n"
        f"최종 입론은 명확한 주장, 근거, 예상 반론 대응을 포함하여 완성된 형태로 작성하세요."
    )


def _run_tool_calling_loop(messages: List) -> str:
    """도구 실행 루프: tool_calls가 없을 때까지 모델 ↔ 도구를 반복 호출한다.

    Args:
        messages: [SystemMessage, HumanMessage, ...] 초기 메시지 리스트 (in-place 확장됨)

    Returns:
        최종 평문 답변(입론) 텍스트
    """
    while True:
        response: AIMessage = _llm_with_tools.invoke(messages)
        messages.append(response)

        # tool_calls가 없으면 최종 답변 반환
        if not response.tool_calls:
            return response.content if isinstance(response.content, str) else str(response.content)

        # 각 tool_call을 실행하고 ToolMessage로 결과를 주입
        for tc in response.tool_calls:
            tool_name: str = tc["name"]
            tool_args: Dict = tc["args"]
            tool_call_id: str = tc["id"]

            if tool_name in _TOOL_MAP:
                tool_result: str = _TOOL_MAP[tool_name].invoke(tool_args)
            else:
                tool_result = f"[오류] 알 수 없는 도구: {tool_name}"

            messages.append(
                ToolMessage(content=tool_result, tool_call_id=tool_call_id)
            )


# ── 메인 노드 ─────────────────────────────────────────────────────────────────

def opening_arguments_node(state: DebateState) -> DebateState:
    """Phase 1 입론 노드.

    speaking_order에서 AI 에이전트 순서를 존중하여 각 에이전트가 순차적으로
    입론을 생성한다. 사용자(user)는 이 노드에서 발언하지 않는다.

    Args:
        state: 현재 DebateState (phase == "opening" 을 전제)

    Returns:
        debate_history가 누적되고 phase가 "chained_rebuttal"로 변경된 DebateState
    """
    topic: str = state["topic"]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]

    # agent_id → AgentSnapshot 빠른 조회
    agent_map = {a["agent_id"]: a for a in state["agents"]}

    # speaking_order에서 AI 에이전트만 추출 (user 제외, 순서 유지)
    ai_speaker_ids = [sid for sid in state["speaking_order"] if sid != "user"]

    print(f"\n[Phase 1] 입론 시작 — 발언 순서: {ai_speaker_ids}\n")

    for speaker_id in ai_speaker_ids:
        agent = agent_map[speaker_id]

        print(f"  [{speaker_id} | {agent['stance']}] 입론 생성 중...")

        # 메시지 구성: 페르소나 주입 + 입론 요청
        messages = [
            SystemMessage(content=agent["system_prompt"]),
            HumanMessage(content=_build_opening_prompt(topic, agent["stance"])),
        ]

        # 도구 실행 루프 → 최종 입론 텍스트
        final_text = _run_tool_calling_loop(messages)

        # DebateHistory에 누적
        entry: DebateEntry = DebateEntry(
            turn=current_turn,
            speaker_id=speaker_id,
            stance=agent["stance"],
            phase="opening",
            content=final_text,
            target_id=None,
        )
        history.append(entry)
        current_turn += 1

        print(f"  [{speaker_id}] 입론 완료 (turn={entry['turn']})\n")

    # 모든 AI 입론 완료 → 다음 단계(2단계 연쇄 논박)로 전환
    print("[Phase 1] 입론 단계 완료 → chained_rebuttal 단계로 전환\n")

    return DebateState(
        **{
            **state,
            "debate_history": history,
            "current_turn": current_turn,
            "current_speaker_index": len(ai_speaker_ids),  # user 직전 위치
            "phase": "chained_rebuttal",
        }
    )
