"""
writer.py — DeepSeek 14B 기반 발언 생성 모듈

역할:
    - LLM 호출 + 후처리 + 재시도 공통 패턴
    - 각 단계별 발언 생성 함수
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import APITimeoutError, APIConnectionError, APIStatusError

logger = logging.getLogger(__name__)

# ── DeepSeek 14B vLLM 클라이언트 ───────────────────────────────────────────
_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

LLM_KWARGS = dict(
    model=os.environ.get("DEEPSEEK_MODEL", "Corianas/DeepSeek-R1-Distill-Qwen-14B-AWQ"),
    base_url=_VLLM_BASE_URL,
    api_key="fake",
    temperature=0.6,
    max_tokens=2048,
    top_p=0.9,
    timeout=120,
)

main_llm = ChatOpenAI(**LLM_KWARGS)
rebuttal_llm = ChatOpenAI(**{**LLM_KWARGS, "max_tokens": 1024})
free_rebuttal_llm = ChatOpenAI(**{**LLM_KWARGS, "max_tokens": 1024, "temperature": 0.6})
synthesis_llm = ChatOpenAI(**{**LLM_KWARGS, "max_tokens": 1024, "temperature": 0.7})
role_reversal_llm = ChatOpenAI(**{**LLM_KWARGS, "max_tokens": 2048, "temperature": 0.6})

_LLM_MAX_RETRIES = 3
_LLM_RETRY_DELAY = 5


# ── 공통 유틸리티 ──────────────────────────────────────────────────────────

def invoke_with_retry(llm, messages: list, *, label: str = "llm") -> AIMessage:
    """타임아웃/연결 오류 시 최대 3회 재시도."""
    for attempt in range(1, _LLM_MAX_RETRIES + 1):
        try:
            return llm.invoke(messages)
        except (APITimeoutError, APIConnectionError) as e:
            logger.warning("[%s] 시도 %d/%d 실패: %s", label, attempt, _LLM_MAX_RETRIES, e)
            if attempt == _LLM_MAX_RETRIES:
                raise
            time.sleep(_LLM_RETRY_DELAY * attempt)
        except APIStatusError as e:
            if e.status_code >= 500:
                logger.warning("[%s] 서버 오류 %d, 재시도 %d/%d", label, e.status_code, attempt, _LLM_MAX_RETRIES)
                if attempt == _LLM_MAX_RETRIES:
                    raise
                time.sleep(_LLM_RETRY_DELAY * attempt)
            else:
                raise


def truncate_tool_result(result: str, max_chars: int = 800) -> str:
    """검색 결과를 최대 길이로 자른다."""
    if len(result) <= max_chars:
        return result
    return result[:max_chars] + "\n[일부만 표시]"
