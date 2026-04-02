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
import os
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

_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

_LLM_KWARGS = dict(
    model="Qwen/Qwen3.5-9B",
    base_url=_VLLM_BASE_URL,
    api_key="fake",          # vLLM은 API 키 불필요, 빈값 아닌 임의값 필요
    temperature=0.7,
)

# 도구 호출용 (일반 모드)
_llm = ChatOpenAI(**_LLM_KWARGS)
_llm_with_tools = _llm.bind_tools(_TOOLS)

# 최종 발언 생성용 (JSON 강제 모드 — vLLM 토큰 레벨에서 JSON 구조 보장)
_llm_json = ChatOpenAI(
    **_LLM_KWARGS,
    model_kwargs={"response_format": {"type": "json_object"}},
)


# ── 응답 후처리 유틸리티 ───────────────────────────────────────────────────────

def _postprocess_speech(text: str) -> str:
    """speech 후처리: 구조 강제 + 볼드 정규화 + 외국 문자 제거."""
    # 1. 제목 정규화: #로 시작하는 줄의 prefix를 '## '로 통일
    text = re.sub(r'^#+[^가-힣a-zA-Z0-9\n]*(?=[가-힣a-zA-Z])', '## ', text, flags=re.MULTILINE)
    # 2. 소제목(## 로 시작하는 줄)에서 볼드 마크다운(*, **) 제거
    text = re.sub(r'^(## .*)$', lambda m: m.group(1).replace('*', ''), text, flags=re.MULTILINE)
    # 3. 분석 라벨 제거 (원인:, 메커니즘:, 결과: — 볼드 포함)
    text = re.sub(r'\*{0,2}(?:원인|메커니즘|결과)\*{0,2}\s*[:：]\s*', '', text)
    # 4. 한자·일본어 등 외국 문자 제거
    text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff]+', '', text)
    text = re.sub(r' {2,}', ' ', text)
    # 5. 논거 3 이후 강제 잘라내기 (## 결론은 보존)
    match = re.search(r'^## 논거\s*3', text, re.MULTILINE)
    if match:
        # 논거 3 시작부터 ## 결론 직전까지 제거
        conclusion = re.search(r'^## 결론', text[match.start():], re.MULTILINE)
        if conclusion:
            text = text[:match.start()] + text[match.start() + conclusion.start():]
        else:
            # 결론이 없으면 논거 3부터 끝까지 제거
            text = text[:match.start()]
    # 6. 볼드 마크다운 정규화: ****(4개), ***(3개) → **(2개)
    text = re.sub(r'\*{3,}([^*]+?)\*{3,}', r'**\1**', text)
    # 7. 빈 볼드(****, ** ** 등) 제거
    text = re.sub(r'\*{2,}\s*\*{2,}', '', text)
    return text


def _extract_speech_from_json(content: str) -> Tuple[str, str]:
    """JSON 응답에서 speech 필드를 추출하고, 제목 레벨을 통일하여 반환한다.

    Returns:
        (speech, json_raw) — speech는 후처리된 발언, json_raw는 LLM 원본 응답
    """
    json_raw = content.strip()
    text = json_raw

    # 1. <think> 블록 제거 (JSON 모드에서도 CoT가 나올 수 있음)
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = text.replace('</think>', '').strip()

    # 2. JSON 파싱 → speech 추출
    try:
        data = json.loads(text)
        if isinstance(data, dict) and "speech" in data:
            speech = data["speech"]
            logger.info("[CoT reasoning] %s", data.get("reasoning", "")[:100])
            return _postprocess_speech(speech), json_raw
    except (json.JSONDecodeError, TypeError):
        pass

    # 3. 폴백: JSON 블록 추출 재시도
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, dict) and "speech" in data:
                return _postprocess_speech(data["speech"]), json_raw
        except (json.JSONDecodeError, TypeError):
            pass

    # 4. 최종 폴백: <tool_call> 제거 후 원본 반환
    logger.warning("[_extract_speech_from_json] JSON 파싱 실패, 원본 텍스트 반환")
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    return _postprocess_speech(text.strip()), json_raw


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

def _build_opening_prompt(topic: str, stance: str, focus_area: str, stance_num: int) -> str:
    """입론 요청 HumanMessage 본문을 생성한다."""
    stance_kr = "찬성(PRO)" if stance == "PRO" else "반대(CON)"
    stance_label = "찬성" if stance == "PRO" else "반대"
    agent_name = f"{stance_label} 에이전트{stance_num}"
    return f"""search_web과 search_vector_db를 호출해 근거를 수집한 뒤, '{topic}'에 대한 {stance_kr} 입론을 작성하세요.

[출력 형식]
반드시 아래 JSON 형태로만 응답하세요:
{{"reasoning": "검색 결과 분석, 논리 구성 계획 (이 부분은 관중에게 보이지 않습니다)", "speech": "아래 구조를 따르는 최종 토론 발언"}}

speech는 마크다운 형식으로 작성하세요. ## 소제목 뒤에는 반드시 줄바꿈 후 본문을 작성하세요.
**강조 표시**는 핵심적인 문장에 사용하되, 남용하지 마세요.
상대는 고등학생입니다. 따라서 speech의 내용은 고등학생 수준에 맞게 간결하게 작성하세요.

speech의 구조:
## 인사말
자신의 이름은 "{agent_name}"이며, 논제에 대한 입장을 간결하게 밝힌다.
## 입장 표명
핵심 주장을 한 문장으로 명확하게 선언한다.
## 논거 1: (소제목)
원인 → 메커니즘 → 결과 순서로 토론 근거 대본을 전개한다.
## 논거 2: (소제목)
다른 각도에서 주장을 보강하거나, 상대방 예상 반론에 선제 대응한다.
## 결론
논거를 종합하고, 핵심 주장을 힘 있게 재확인한다.

[진영 고수 규칙]
1. 스탠스 고정: 무조건 {stance_kr} 입장만 방어하라. 상대 진영 논리에 동조하거나 타협하는 것은 절대 금지.
2. 불리한 정보 반박: 검색 결과에 당신의 진영에 불리한 내용이 있다면 절대 수용하지 마라. 반드시 "일각에서는 ~라 우려하지만" 형태의 예상 반론으로 삼아 철저히 논파하라.
3. 결론 일관성: 모든 발언의 마지막은 반드시 {stance_kr} 입장을 강력히 재확인하며 끝내라.

[주의] 한자(漢字), 일본어, 아랍 문자 등 외국 문자 사용 금지. 반드시 한글로만 작성하세요."""


_MAX_TOOL_ROUNDS = 3       # 도구 호출 최대 라운드 수 (무한 루프 방지)
_MAX_TOOL_RESULT_CHARS = 800  # 도구 결과 최대 길이 (컨텍스트 폭발 방지)


def _truncate_tool_result(result: str, max_chars: int = _MAX_TOOL_RESULT_CHARS) -> str:
    """도구 결과가 너무 길면 잘라낸다."""
    if len(result) <= max_chars:
        return result
    return result[:max_chars] + "\n\n[결과가 길어 일부만 표시됩니다]"


def _run_tool_calling_loop(messages: List) -> Tuple[str, List[Dict]]:
    """도구 실행 루프: tool_calls가 없을 때까지 모델 ↔ 도구를 반복 호출한다.

    [처리 순서]
    1. 모델 호출 후 response.tool_calls(vLLM 정식 파싱) 확인
    2. 비어 있으면 content 내 <tool_call> XML 폴백 파싱 (Qwen3.5 호환)
    3. 도구가 없으면 최종 답변으로 판단 → <think>/<tool_call> 블록 제거 후 반환
    4. 도구가 있으면 실행 후 결과를 메시지에 주입하고 반복
    5. 최대 라운드 수 초과 시 강제로 최종 발언 생성으로 전환

    Args:
        messages: [SystemMessage, HumanMessage, ...] 초기 메시지 리스트 (in-place 확장됨)

    Returns:
        (순수 입론 텍스트, JSON 원본 응답, 사용된 도구 로그 리스트)
    """
    tool_calls_log: List[Dict] = []
    round_count = 0

    while True:
        response: AIMessage = _llm_with_tools.invoke(messages)
        content: str = response.content if isinstance(response.content, str) else str(response.content)

        # ── 도구 호출 감지: 정식 파싱 우선, 없으면 XML 폴백 ──────────────────
        tool_calls = list(response.tool_calls) if response.tool_calls else _parse_xml_tool_calls(content)

        if not tool_calls or round_count >= _MAX_TOOL_ROUNDS:
            # 더 이상 도구 호출 없음 (또는 최대 라운드 초과) → JSON 모드로 최종 발언 생성
            if round_count >= _MAX_TOOL_ROUNDS and tool_calls:
                logger.info("[tool_loop] 최대 도구 호출 라운드(%d) 도달, 최종 발언 생성으로 전환", _MAX_TOOL_ROUNDS)
            messages.append(response)
            messages.append(HumanMessage(
                content='위 검색 결과를 바탕으로 입론을 작성하세요. '
                        '반드시 {"reasoning": "분석 과정", "speech": "최종 발언"} JSON으로만 출력하세요.'
            ))
            json_response: AIMessage = _llm_json.invoke(messages)
            json_content: str = json_response.content if isinstance(json_response.content, str) else str(json_response.content)
            speech, json_raw = _extract_speech_from_json(json_content)
            return speech, json_raw, tool_calls_log

        # ── 도구 호출 로그 수집 ───────────────────────────────────────────────
        for tc in tool_calls:
            tool_calls_log.append({"name": tc["name"], "args": tc["args"]})

        round_count += 1

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
                messages.append(ToolMessage(content=_truncate_tool_result(tool_result), tool_call_id=tc["id"]))
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
                results.append(f"[{tc['name']} 결과]\n{_truncate_tool_result(tool_result)}")
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

    # agent_id → AgentSnapshot 빠른 조회
    agent_map = {a["agent_id"]: a for a in state["agents"]}

    # 진영별 번호 카운터 (찬성 에이전트1~3, 반대 에이전트1~3)
    _stance_counter: Dict[str, int] = {"PRO": 0, "CON": 0}

    print(f"\n[1단계: 입론] 발언 순서: {state['speaking_order']}\n")

    for idx, speaker_id in enumerate(state["speaking_order"]):
        if speaker_id == "user":
            continue  # 사용자 턴은 API를 통해 별도 처리

        agent = agent_map[speaker_id]
        _stance_counter[agent["stance"]] += 1
        _stance_num = _stance_counter[agent["stance"]]
        _stance_label = "찬성" if agent["stance"] == "PRO" else "반대"
        _display_name = f"{_stance_label} 에이전트{_stance_num}"

        print(f"  [{_display_name}] 입론 생성 중...")

        # 메시지 구성: 페르소나 주입 + 입론 요청
        messages = [
            SystemMessage(content=agent["system_prompt"]),
            HumanMessage(content=_build_opening_prompt(topic, agent["stance"], agent["focus_area"], _stance_num)),
        ]

        # 도구 실행 루프 → 최종 입론 텍스트 + JSON 원본 + 도구 사용 로그
        final_text, json_raw, tool_calls_log = _run_tool_calling_loop(messages)

        # DebateHistory에 누적
        entry: DebateEntry = DebateEntry(
            turn=idx,
            speaker_id=speaker_id,
            stance=agent["stance"],
            phase="opening",
            content=final_text,
            target_id=None,
            tool_calls_log=tool_calls_log,
            json_raw=json_raw,
        )
        history.append(entry)

        print(f"  [{_display_name}] 입론 완료 (turn={entry['turn']})\n")

    # history를 turn 기준으로 정렬 (speaking_order 순서 보장)
    history.sort(key=lambda e: e["turn"])

    # 모든 AI 입론 완료 → 사용자 입론 대기
    print("[1단계: 입론] AI 에이전트 입론 완료 → 사용자 입론 대기\n")

    return DebateState(
        **{
            **state,
            "debate_history": history,
            "current_turn": len(state["speaking_order"]),
            "current_speaker_index": len(state["speaking_order"]),
            "phase": "opening",  # 사용자 입론 완료 전까지 phase 유지
        }
    )
