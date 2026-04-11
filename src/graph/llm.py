"""
llm.py — Qwen2.5-32B LLM 클라이언트 + search_web 도구

단일 모델로 생성+분석+검색 판단을 모두 처리.
"""

from __future__ import annotations

import logging
import os
import time

from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import APITimeoutError, APIConnectionError, APIStatusError
from tavily import TavilyClient

logger = logging.getLogger(__name__)

# ── .env 로드 ──────────────────────────────────────────────────────────────
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            if _line.strip().startswith("TAVILY_API_KEY="):
                os.environ["TAVILY_API_KEY"] = _line.strip().split("=", 1)[1]
                break

# ── Tavily 검색 도구 ──────────────────────────────────────────────────────
_tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))


@tool
def search_web(query: str) -> str:
    """웹에서 최신 뉴스 및 정보를 검색합니다. 반박에 통계나 사실 확인이 필요할 때 사용하세요."""
    try:
        results = _tavily_client.search(
            query,
            max_results=3,
            search_depth="basic",
            exclude_domains=[
                "blog.naver.com", "m.blog.naver.com",
                "tistory.com", "brunch.co.kr",
                "linkedin.com", "medium.com",
                "velog.io", "daum.net",
            ],
        )
        items = results.get("results", [])
        if not items:
            return "[검색 결과] 관련 결과를 찾을 수 없습니다."
        return "[검색 결과]\n" + "\n".join(
            f"- {r['title']}: {r['content'][:200]}" for r in items
        )
    except Exception as e:
        return f"[검색 오류] {e}"


# ── LLM 클라이언트 ────────────────────────────────────────────────────────
_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

_LLM_KWARGS = dict(
    model=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ"),
    base_url=_VLLM_BASE_URL,
    api_key="fake",
    temperature=0.6,
    max_tokens=2048,
    top_p=0.9,
    timeout=180,  # 32B는 좀 더 오래 걸림
)

llm = ChatOpenAI(**_LLM_KWARGS)
llm_with_tools = llm.bind_tools([search_web])

# 종합 회의용 (짧은 응답)
llm_short = ChatOpenAI(**{**_LLM_KWARGS, "max_tokens": 512, "temperature": 0.7})


# ── 공통 유틸 ──────────────────────────────────────────────────────────────

_MAX_RETRIES = 3
_RETRY_DELAY = 5


def invoke_with_retry(target_llm, messages: list, *, label: str = "llm") -> AIMessage:
    """타임아웃/연결 오류 시 최대 3회 재시도."""
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            return target_llm.invoke(messages)
        except (APITimeoutError, APIConnectionError) as e:
            logger.warning("[%s] 시도 %d/%d 실패: %s", label, attempt, _MAX_RETRIES, e)
            if attempt == _MAX_RETRIES:
                raise
            time.sleep(_RETRY_DELAY * attempt)
        except APIStatusError as e:
            if e.status_code >= 500:
                logger.warning("[%s] 서버 오류 %d, 재시도 %d/%d", label, e.status_code, attempt, _MAX_RETRIES)
                if attempt == _MAX_RETRIES:
                    raise
                time.sleep(_RETRY_DELAY * attempt)
            else:
                raise


def invoke_with_tools(messages: list, *, label: str = "llm") -> tuple:
    """tool calling LLM 호출. tool call이 있으면 실행 후 재호출.

    Returns:
        (speech_text, raw_text, tool_calls_log)
    """
    from langchain_core.messages import ToolMessage

    response = invoke_with_retry(llm_with_tools, messages, label=label)
    raw = response.content if isinstance(response.content, str) else str(response.content)
    tool_calls_log = []

    # tool call이 있으면 실행
    if response.tool_calls:
        for tc in response.tool_calls:
            tool_calls_log.append({"name": tc["name"], "args": tc["args"]})
            result = search_web.invoke(tc["args"])
            tool_calls_log[-1]["result"] = result[:200]

            messages.append(response)
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        # 도구 결과 포함하여 재호출 (도구 없이)
        final = invoke_with_retry(llm, messages, label=f"{label}_final")
        raw = final.content if isinstance(final.content, str) else str(final.content)

    return raw, tool_calls_log


# ── 공통 히스토리 체인 빌더 ────────────────────────────────────────────────

def build_debate_chain(
    history: list,
    current_agent_id: str,
    phase_filter: str = "",
) -> list:
    """debate_history를 LLM 메시지 체인으로 변환한다.

    - 현재 에이전트 발언 → AIMessage
    - 다른 발언자 → HumanMessage (speaker 라벨 포함)

    Args:
        history: debate_history 리스트
        current_agent_id: 현재 발언하는 에이전트 ID
        phase_filter: 특정 phase만 포함 ("" = 전체)

    Returns:
        LangChain 메시지 리스트 (SystemMessage 미포함)
    """
    from langchain_core.messages import AIMessage as AI, HumanMessage as HM

    messages = []
    for entry in history:
        if phase_filter and entry.get("phase") != phase_filter:
            continue

        content = entry["content"]
        speaker_id = entry["speaker_id"]
        stance = "찬성" if entry.get("stance") == "PRO" else "반대"

        if speaker_id == current_agent_id:
            messages.append(AI(content=content))
        elif speaker_id == "user":
            messages.append(HM(content=f"[사용자 ({stance})] {content}"))
        else:
            messages.append(HM(content=f"[{speaker_id} ({stance})] {content}"))

    return messages
