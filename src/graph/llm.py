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
    # 환경변수 SEARCH_CACHE_LOOKUP=0 이면 캐시 조회 비활성 (순수 Tavily + 저장만)
    _lookup_enabled = os.environ.get("SEARCH_CACHE_LOOKUP", "1").strip() not in ("0", "false", "False", "")

    # 컨텍스트에서 이미 사용된(중복 금지) URL 가져오기
    try:
        from src.graph.vector_store import get_excluded_urls, mark_urls_used
        excluded = get_excluded_urls()
    except Exception:
        excluded = set()
        mark_urls_used = None

    # 1) Pinecone 의미 캐시 조회 — hit 시 Tavily 호출 없이 반환
    if _lookup_enabled:
        try:
            from src.graph.vector_store import semantic_cache_lookup, format_cached_results
            cached = semantic_cache_lookup(query)
            if cached:
                logger.info("[search_web] 캐시 hit (top=%.3f, %d건, 중복제외=%d)",
                            cached[0]["score"], len(cached), len(excluded))
                if mark_urls_used:
                    mark_urls_used(r.get("url", "") for r in cached)
                return format_cached_results(cached)
        except Exception as e:
            logger.warning("[search_web] 캐시 조회 실패, Tavily로 폴백: %s", e)

    # 2) 캐시 miss 또는 lookup 비활성 → Tavily 호출
    try:
        from src.graph.vector_store import upsert_search_results
        results = _tavily_client.search(
            query,
            max_results=5,  # 필터링 감안해 여유 있게
            search_depth="advanced",
            exclude_domains=[
                "blog.naver.com", "m.blog.naver.com",
                "tistory.com", "brunch.co.kr",
                "linkedin.com", "medium.com",
                "velog.io", "daum.net",
            ],
        )
        items = results.get("results", [])
        # 이미 사용된 URL 제외
        items = [r for r in items if r.get("url") not in excluded][:3]
        if not items:
            return "[검색 결과] 관련 결과를 찾을 수 없습니다."
        # 3) Pinecone 저장 — 기본 OFF (벤치마크 인덱스 오염 방지).
        # 명시적으로 SEARCH_CACHE_SAVE=1 설정 시에만 저장 (인덱스 구축/업데이트 작업 시)
        _save_enabled = os.environ.get("SEARCH_CACHE_SAVE", "0").strip() in ("1", "true", "True")
        if _save_enabled:
            try:
                upsert_search_results(query, items)
            except Exception as e:
                logger.warning("[search_web] 캐시 저장 실패: %s", e)
        # 사용된 URL 누적 (다음 검색에서 중복 제외)
        if mark_urls_used:
            mark_urls_used(r.get("url", "") for r in items)
        return "[검색 결과]\n" + "\n".join(
            f"- {r['content'][:300]}" for r in items
        )
    except Exception as e:
        return f"[검색 오류] {e}"


def _normalize_tool_args(args) -> dict:
    """모델이 중첩 구조({'arguments': {'query': X}})로 인자를 생성해도 평탄화한다."""
    if not isinstance(args, dict):
        return {"query": str(args)}
    # 중첩된 'arguments' 키만 있는 경우 풀기
    if "arguments" in args and isinstance(args["arguments"], dict) and "query" not in args:
        args = args["arguments"]
    # query 키 없으면 첫 문자열 값으로 대체
    if "query" not in args:
        for v in args.values():
            if isinstance(v, str) and v.strip():
                return {"query": v}
        return {"query": ""}
    return args


def safe_search_invoke(args) -> str:
    """search_web 호출 안전 래퍼. 인자 구조 정규화 + 예외 포착."""
    try:
        normalized = _normalize_tool_args(args)
        if not normalized.get("query", "").strip():
            return "[검색 실패] 빈 쿼리"
        return search_web.invoke(normalized)
    except Exception as e:
        logger.warning("[safe_search_invoke] 실패: %s (args=%r)", e, args)
        return f"[검색 실패] {e}"


# ── LLM 클라이언트 ────────────────────────────────────────────────────────
_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

_LLM_KWARGS = dict(
    model=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ"),
    base_url=_VLLM_BASE_URL,
    api_key=os.environ.get("LLM_API_KEY", "fake"),
    temperature=0.6,
    max_tokens=2048,
    top_p=0.9,
    timeout=180,  # 32B는 좀 더 오래 걸림
)

llm = ChatOpenAI(**_LLM_KWARGS)
llm_with_tools = llm.bind_tools([search_web])

# 종합 회의용 (짧은 응답)
llm_short = ChatOpenAI(**{**_LLM_KWARGS, "max_tokens": 512, "temperature": 0.7})

# 어시스턴트(DebateAssistant) 용 — 자유논박처럼 prompt+history 가 길어
# 내부 추론 후 출력 분량이 모자라 잘리는 사례가 있어 한도 상향.
llm_assistant = ChatOpenAI(**{**_LLM_KWARGS, "max_tokens": 3072})
llm_assistant_with_tools = llm_assistant.bind_tools([search_web])


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


def invoke_with_tools(
    messages: list,
    *,
    label: str = "llm",
    tools_llm=None,
    final_llm=None,
) -> tuple:
    """tool calling LLM 호출. tool call이 있으면 실행 후 재호출.

    Parameters
    ----------
    tools_llm : tool 바인딩된 ChatOpenAI. None 이면 기본 `llm_with_tools` 사용.
    final_llm : tool 결과 받은 뒤 최종 출력용 ChatOpenAI. None 이면 기본 `llm` 사용.
        어시스턴트처럼 더 큰 max_tokens 가 필요한 경우 override.

    Returns:
        (speech_text, raw_text, tool_calls_log)
    """
    from langchain_core.messages import ToolMessage

    _tools_llm = tools_llm if tools_llm is not None else llm_with_tools
    _final_llm = final_llm if final_llm is not None else llm

    response = invoke_with_retry(_tools_llm, messages, label=label)
    raw = response.content if isinstance(response.content, str) else str(response.content)
    tool_calls_log = []

    # tool call이 있으면 실행
    if response.tool_calls:
        for tc in response.tool_calls:
            tool_calls_log.append({"name": tc["name"], "args": tc["args"]})
            result = safe_search_invoke(tc["args"])
            tool_calls_log[-1]["result"] = result[:200]

            messages.append(response)
            messages.append(ToolMessage(content=result, tool_call_id=tc["id"]))

        # 도구 결과 포함하여 재호출 (도구 없이)
        final = invoke_with_retry(_final_llm, messages, label=f"{label}_final")
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
