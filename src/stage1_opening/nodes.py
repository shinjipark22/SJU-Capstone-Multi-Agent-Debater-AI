"""
nodes.py — 1단계: 입론(Opening Arguments) 노드

[동작 흐름]
    1. speaking_order에서 AI 에이전트(user 제외)만 순서대로 추출
    2. 각 에이전트마다:
        a. system_prompt → SystemMessage, 입론 요청 → HumanMessage 구성
        b. LLM + 도구 실행 루프: tool_calls가 없어질 때까지 ToolMessage 주입 반복
        c. 최종 텍스트(입론)를 DebateEntry 형태로 debate_history에 누적
    3. 모든 AI 입론 완료 후 phase를 "chained_rebuttal"(2단계 연쇄 논박)로 전환

[설계 노트]
    - 도구 실행 루프는 각 에이전트 호출마다 독립적으로 동작 (메시지 격리)
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from ddgs import DDGS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from src.stage1_opening.vector_db import query_vector_db
from src.state import DebateEntry, DebateState


# ── 세션 중복 문서 추적 (에이전트 간 동일 문서 중복 인용 방지) ─────────────────────
# opening_arguments_node 진입 시 초기화되며, search_vector_db 호출마다 갱신된다.
_used_doc_ids: set = set()


# ── 도구 정의 ─────────────────────────────────────────────────────────────────

@tool
def search_web(query: str) -> str:
    """웹에서 최신 뉴스 및 정보를 DuckDuckGo로 검색합니다.

    Args:
        query: 검색할 키워드 또는 질문
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "[웹 검색 결과] 관련 결과를 찾을 수 없습니다."
        summaries = [
            f"- {r['title']}: {r['body']}"
            for r in results
        ]
        return "[웹 검색 결과]\n" + "\n".join(summaries)
    except Exception as e:
        return f"[웹 검색 오류] {e}"


@tool
def search_vector_db(query: str, topic: str, stance: str) -> str:
    """토론 전문가 문서 VectorDB에서 진영·주제 필터링을 적용하여 관련 근거를 검색합니다.
    이미 다른 에이전트가 인용한 문서는 자동으로 제외하여 중복 인용을 방지합니다.

    Args:
        query:  검색할 내용 (자연어 질의)
        topic:  현재 토론 주제 (메타데이터 필터)
        stance: 검색할 진영 "PRO" 또는 "CON" (메타데이터 필터)
    """
    try:
        docs, ids = query_vector_db(
            query=query,
            topic=topic,
            stance=stance,
            exclude_ids=list(_used_doc_ids),
        )
        if not docs:
            return "[문서 검색 결과] 관련 문서를 찾을 수 없습니다."
        # 반환된 문서 ID를 세션 추적 세트에 등록하여 이후 에이전트가 중복 인용하지 않도록 함
        _used_doc_ids.update(ids)
        return "[문서 검색 결과]\n" + "\n".join(f"- {d}" for d in docs)
    except Exception as e:
        logger.warning("[search_vector_db] 도구 실행 오류: %s", e)
        return "[문서 검색 결과] 관련된 구체적인 데이터를 찾을 수 없습니다. 웹 검색을 시도하세요."


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


# ── 응답 후처리 유틸리티 ───────────────────────────────────────────────────────

def _clean_response(content: str) -> str:
    """순수 입론 텍스트만 반환한다.

    처리 순서:
    1. <think>...</think> 블록 전체 제거 (닫힌 경우)
    2. 닫히지 않은 <think> 이후 내용 전체 제거 (모델이 thinking 중 출력이 끊긴 경우)
    3. 남은 </think> 단독 태그 제거
    4. <tool_call> 블록 제거
    5. CJK 한자 룰베이스 제거
    """
    text = content
    # 1. 닫힌 <think>...</think> 블록 제거
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    # 2. 닫히지 않은 <think> 이후 텍스트 전체 제거
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    # 3. 남은 </think> 단독 태그 제거
    text = text.replace('</think>', '')
    # 4. <tool_call> 블록 제거
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    # 5. CJK 한자 제거 (U+4E00–U+9FFF 등 3개 범위) — Language Leak 안전망
    text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+', '', text)
    # 한자 제거로 생긴 연속 공백 정리
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def _extract_final_speech(text: str) -> str:
    """JSON 응답에서 final_speech만 추출한다.

    모델이 JSON 형식을 지켰으면 final_speech를 꺼내고,
    실패하면 원본 텍스트를 그대로 반환한다 (폴백).
    """
    cleaned = text.strip()
    # 마크다운 코드블록 제거 (모델이 ```json ... ``` 을 붙이는 경우 대비)
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```\s*$', '', cleaned)
    cleaned = cleaned.strip()

    # 1차: 전체 텍스트가 JSON인 경우
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict) and "final_speech" in data:
            return data["final_speech"]
    except (json.JSONDecodeError, TypeError):
        pass

    # 2차: 텍스트 중간에 JSON 블록이 있는 경우
    json_match = re.search(r'\{[^{}]*"final_speech"\s*:\s*".*?"[^{}]*\}', cleaned, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if "final_speech" in data:
                return data["final_speech"]
        except (json.JSONDecodeError, TypeError):
            pass

    # 폴백: JSON 파싱 실패 시 원본 반환
    return text


def _parse_xml_tool_calls(content: str) -> List[Dict]:
    """content 내 <tool_call> XML 블록을 파싱하여 tool_calls 형태로 반환한다.

    Qwen3.5는 vLLM의 hermes 파서가 인식하지 못하는 XML 포맷으로 도구를 호출한다.
    response.tool_calls가 비어 있을 때 이 함수로 폴백 파싱한다.

    출력 형식 예시:
        <tool_call>
        <function=search_web>
        <parameter=query>검색어</parameter>
        </function>
        </tool_call>
    """
    tool_calls = []
    for block in re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL):
        func_match = re.search(r'<function=(\w+)>(.*?)</function>', block, re.DOTALL)
        if not func_match:
            continue
        func_name = func_match.group(1)
        args: Dict[str, str] = {}
        for param in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', func_match.group(2), re.DOTALL):
            args[param.group(1)] = param.group(2).strip()
        tool_calls.append({
            "name": func_name,
            "args": args,
            "id": f"call_{func_name}_{uuid.uuid4().hex[:6]}",
        })
    return tool_calls


# ── 내부 유틸리티 ─────────────────────────────────────────────────────────────

def _build_opening_prompt(topic: str, stance: str, focus_area: str) -> str:
    """입론 요청 HumanMessage 본문을 생성한다."""
    stance_kr = "찬성(PRO)" if stance == "PRO" else "반대(CON)"
    return (
        f"[필수] 반드시 한국어로만 작성하라. 중국어 한자를 단 한 글자도 쓰지 마라.\n\n"
        f"토론 주제: {topic}\n"
        f"당신의 진영: {stance_kr} / 전문 분야: {focus_area}\n\n"
        f"지금 즉시 search_web과 search_vector_db를 호출해 '{focus_area}' 관련 근거를 수집하라.\n"
        f"검색 결과에 실제로 있는 수치와 사례만 사용해 입론을 작성하라. 없는 데이터는 절대 지어내지 마라.\n"
        f"인사말 없이 첫 문장부터 바로 핵심 주장으로 시작하고, 마크다운 기호는 쓰지 마라.\n"
        f"반드시 '~입니다', '~습니다' 체(합쇼체)로만 작성하라.\n"
        f"입론 본문이 끝난 뒤 반드시 아래 형식으로 출처 목록을 첨부하라:\n"
        f"[참고 자료]\n"
        f"1. (기관/저자명, 연도) 자료명 — 인용한 수치 또는 주장 한 줄 요약\n"
        f"검색 결과에 실제로 등장한 자료만 적고, 지어낸 출처는 절대 쓰지 마라."
    )


def _run_tool_calling_loop(messages: List) -> Tuple[str, List[Dict]]:
    """도구 실행 루프: tool_calls가 없을 때까지 모델 ↔ 도구를 반복 호출한다.

    [처리 순서]
    1. 모델 호출 후 response.tool_calls(vLLM 정식 파싱) 확인
    2. 비어 있으면 content 내 <tool_call> XML 폴백 파싱 (Qwen3.5 호환)
    3. 도구가 없으면 최종 답변으로 판단 → <think>/<tool_call> 블록 제거 후 반환
    4. 도구가 있으면 실행 후 결과를 메시지에 주입하고 반복

    Args:
        messages: [SystemMessage, HumanMessage, ...] 초기 메시지 리스트 (in-place 확장됨)

    Returns:
        (순수 입론 텍스트, 사용된 도구 로그 리스트)
        도구 로그 항목: {"name": str, "args": dict}
    """
    tool_calls_log: List[Dict] = []

    while True:
        response: AIMessage = _llm_with_tools.invoke(messages)
        content: str = response.content if isinstance(response.content, str) else str(response.content)

        # ── 도구 호출 감지: 정식 파싱 우선, 없으면 XML 폴백 ──────────────────
        tool_calls = list(response.tool_calls) if response.tool_calls else _parse_xml_tool_calls(content)

        if not tool_calls:
            # 더 이상 도구 호출 없음 → 최종 답변 반환
            # 1) <think> / CJK 등 정리 → 2) JSON에서 final_speech 추출
            cleaned = _clean_response(content)
            final = _extract_final_speech(cleaned)
            return final, tool_calls_log

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
                content="\n\n".join(results) + "\n\n위 검색 결과를 바탕으로 입론을 완성하세요."
            ))


# ── 메인 노드 ─────────────────────────────────────────────────────────────────

def opening_arguments_node(state: DebateState) -> DebateState:
    """1단계 입론 노드.

    speaking_order에서 AI 에이전트 순서를 존중하여 각 에이전트가 순차적으로
    입론을 생성한다. 사용자(user)는 이 노드에서 발언하지 않는다.

    Args:
        state: 현재 DebateState (phase == "opening" 을 전제)

    Returns:
        debate_history가 누적되고 phase가 "chained_rebuttal"(2단계 연쇄 논박)로 변경된 DebateState
    """
    global _used_doc_ids
    _used_doc_ids = set()  # 새 토론 세션 시작 시 문서 추적 초기화

    topic: str = state["topic"]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]

    # agent_id → AgentSnapshot 빠른 조회
    agent_map = {a["agent_id"]: a for a in state["agents"]}

    # speaking_order에서 AI 에이전트만 추출 (user 제외, 순서 유지)
    ai_speaker_ids = [sid for sid in state["speaking_order"] if sid != "user"]

    print(f"\n[1단계: 입론] 발언 순서: {ai_speaker_ids}\n")

    for speaker_id in ai_speaker_ids:
        agent = agent_map[speaker_id]

        print(f"  [{speaker_id} | {agent['stance']}] 입론 생성 중...")

        # 메시지 구성: 페르소나 주입 + 입론 요청
        messages = [
            SystemMessage(content=agent["system_prompt"]),
            HumanMessage(content=_build_opening_prompt(topic, agent["stance"], agent["focus_area"])),
        ]

        # 도구 실행 루프 → 최종 입론 텍스트 + 도구 사용 로그
        final_text, tool_calls_log = _run_tool_calling_loop(messages)

        # DebateHistory에 누적
        entry: DebateEntry = DebateEntry(
            turn=current_turn,
            speaker_id=speaker_id,
            stance=agent["stance"],
            phase="opening",
            content=final_text,
            target_id=None,
            tool_calls_log=tool_calls_log,
        )
        history.append(entry)
        current_turn += 1

        print(f"  [{speaker_id}] 입론 완료 (turn={entry['turn']})\n")

    # 모든 AI 입론 완료 → 2단계 연쇄 논박으로 전환
    print("[1단계: 입론] 완료 → 2단계 연쇄 논박(chained_rebuttal)으로 전환\n")

    return DebateState(
        **{
            **state,
            "debate_history": history,
            "current_turn": current_turn,
            "current_speaker_index": len(ai_speaker_ids),  # user 직전 위치
            "phase": "chained_rebuttal",
        }
    )
