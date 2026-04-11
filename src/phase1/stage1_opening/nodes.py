"""
nodes.py — 1단계: 입론(Opening Arguments) 노드

[설계 노트]
    - JSON/reasoning 출력 제거, delimiter 기반 자연어 출력
    - 모델은 내부 CoT 후 최종 발언만 출력
    - DeepSeek-R1-Distill-Qwen-14B 최적화
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from openai import APITimeoutError, APIConnectionError, APIStatusError
from pydantic import ValidationError
from tavily import TavilyClient

from src.state import DebateEntry, DebateState


# ── 세션 중복 문서 추적 ─────────────────────────────────────────────────────────
_used_doc_ids: set = set()


# ── 도구 정의 ─────────────────────────────────────────────────────────────────

# .env 파일에서 TAVILY_API_KEY 로드
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            if _line.strip().startswith("TAVILY_API_KEY="):
                os.environ["TAVILY_API_KEY"] = _line.strip().split("=", 1)[1]
                break

_tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))


@tool
def search_web(query: str) -> str:
    """웹에서 최신 뉴스 및 정보를 검색합니다.

    Args:
        query: 검색할 키워드
    """
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





# ── 도구·LLM 초기화 ──────────────────────────────────────────────────────────

_TOOLS: List = [search_web]
_TOOL_MAP: Dict[str, Any] = {t.name: t for t in _TOOLS}

_VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")

_LLM_KWARGS = dict(
    model=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ"),
    base_url=_VLLM_BASE_URL,
    api_key="fake",
    temperature=0.6,
    max_tokens=2048,
    top_p=0.9,
    timeout=120,
)

_llm = ChatOpenAI(**_LLM_KWARGS)
_llm_with_tools = _llm.bind_tools(_TOOLS)

_LLM_MAX_RETRIES = 3
_LLM_RETRY_DELAY = 5


def _invoke_with_retry(llm, messages: list, *, label: str = "llm") -> AIMessage:
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


# ── delimiter 기반 텍스트 추출 + 후처리 ──────────────────────────────────────

def _remove_english_blocks(text: str) -> str:
    """영어로 된 CoT 블록을 제거한다. DeepSeek-R1이 <think> 없이 영어로 사고하는 경우 대응."""
    lines = text.split('\n')
    result = []
    eng_streak = 0
    buffer = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            buffer.append(line)
            continue
        # 한글이 포함된 줄인지 확인
        has_korean = bool(re.search(r'[가-힣]', stripped))
        # 영어/숫자/기호만으로 된 줄
        is_english = not has_korean and bool(re.search(r'[a-zA-Z]', stripped))

        if is_english:
            eng_streak += 1
            buffer.append(line)
        else:
            if eng_streak >= 3:
                # 영어 3줄 이상 연속 → CoT로 판단, buffer 버림
                buffer.clear()
            else:
                result.extend(buffer)
                buffer.clear()
            eng_streak = 0
            result.append(line)

    # 남은 buffer (마지막이 영어 블록이면 버림)
    if eng_streak < 3:
        result.extend(buffer)

    return '\n'.join(result).strip()


def _extract_delimited_text(content: str) -> str:
    """'### 답변 시작' ~ '### 답변 끝' 사이 텍스트를 추출한다."""
    text = content.strip()

    # <think> 블록 제거
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = text.replace('</think>', '').strip()

    # 영어 CoT 블록 제거
    text = _remove_english_blocks(text)

    # delimiter 추출
    m = re.search(r'###\s*답변\s*시작\s*(?:###)?\s*\n?(.*?)\n?\s*###\s*답변\s*끝', text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()

    # '### 답변 시작' 이후 전체
    m = re.search(r'###\s*답변\s*시작\s*(?:###)?\s*\n?(.*)', text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()

    # fallback: 한국어 부분만 추출
    logger.warning("[_extract_delimited_text] delimiter 없음, 한국어 부분만 추출")
    text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
    text = re.sub(r'\{[^}]*\}', '', text, flags=re.DOTALL)
    # 메타 문장 제거
    text = re.sub(r'^.*(?:followed by|then the|end with).*$', '', text, flags=re.MULTILINE)
    return text.strip()


def _postprocess_speech(text: str) -> str:
    """후처리: 제목 정규화 + 외국 문자/영어 문장 제거 + 도구 흔적 제거."""
    # 제목 정규화
    text = re.sub(r'^#+[^가-힣a-zA-Z0-9\n]*(?=[가-힣a-zA-Z])', '### ', text, flags=re.MULTILINE)
    text = re.sub(r'^(### .*)$', lambda m: m.group(1).replace('*', ''), text, flags=re.MULTILINE)
    # 깨진 유니코드 문자 제거
    text = text.replace('\ufffd', '')
    # 외국 문자 + 중국어 문장부호 제거
    text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff\u0400-\u04ff\u0e00-\u0e7f\u0600-\u06ff\u0100-\u024f\u1e00-\u1eff\u00c0-\u00ff\u0150-\u017f\u3000-\u303f\uff00-\uff60]+', '', text)
    # 한국어 문장 중간의 영어 소문자 단어 제거 (대문자 시작 고유명사 McKinsey, OECD 등은 유지)
    text = re.sub(r'(?<=[가-힣\s])[a-z]{5,}(?=[가-힣\s.,])', '', text)
    # 영어 줄 제거 (한글 없이 영어로만 이루어진 줄)
    lines = text.split('\n')
    text = '\n'.join(l for l in lines if not l.strip() or re.search(r'[가-힣]', l) or l.strip().startswith('###'))
    # 영어 고유명사/기관명은 유지, 혼종단어와 영어 전용 줄만 제거
    # 도구명 흔적 제거
    text = re.sub(r'search_web', '', text)
    text = re.sub(r'를 통해 확인되는 자료에 따르면[,.]?\s*', '', text)
    text = re.sub(r'를 통해 (?:최근|확인)', '', text)
    # 영어 잔해 정리 (Forum → 세계경제포럼 등)
    text = text.replace('Forum의', '세계경제포럼의')
    text = text.replace('Forum ', '세계경제포럼 ')
    text = re.sub(r'Naver Blog에 따르면[,.]?\s*', '', text)
    text = re.sub(r'[a-zA-Z]+\s*Blog에 따르면[,.]?\s*', '', text)
    text = re.sub(r'네이버\s*블로그에서\s*언급된\s*바와\s*같이[,.]?\s*', '', text)
    text = re.sub(r'Daum의\s*보도에\s*따르면[,.]?\s*', '', text)
    text = re.sub(r'블로그에\s*따르면[,.]?\s*', '', text)
    # "의장은" → 앞에 이름 없으면 제거
    text = re.sub(r'(?<![가-힣a-zA-Z])의장은\s*', '', text)
    # 영어 단어/혼종단어 제거는 프롬프트로 제어 (후처리에서 삭제 시 구멍 발생)
    # 목록 형태 제거 (- 로 시작하는 줄 → 일반 문장으로)
    text = re.sub(r'^-\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\*\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s+', '', text, flags=re.MULTILINE)
    # "검색 결과" → "관련 분석"으로 교체
    text = text.replace('검색 결과', '관련 분석')
    text = text.replace('검색결과', '관련 분석')
    text = text.replace('자료 조사', '관련 분석')
    # 프롬프트 형식 유출 제거
    text = re.sub(r'\*{0,2}핵심\s*주장\*{0,2}\s*[:：]?\s*', '', text)
    text = re.sub(r'\*{0,2}입장\s*재확인\*{0,2}\s*[:：]?\s*', '', text)
    # 메타 표현 제거
    text = re.sub(r'의 의견을 들어본다[.]?\s*', '은 ', text)
    # 분석 라벨 제거
    text = re.sub(r'\*{0,2}(?:원인|메커니즘|결과)\*{0,2}\s*[:：]\s*', '', text)
    # 볼드 정규화
    text = re.sub(r'\*{3,}([^*]+?)\*{3,}', r'**\1**', text)
    text = re.sub(r'\*{2,}\s*\*{2,}', '', text)
    # 다중 공백 정리
    text = re.sub(r' {2,}', ' ', text)
    # '자기소개와 입장 표명' 중복 텍스트 제거 (소제목 아닌 본문의 동일 텍스트)
    text = re.sub(r'^자기소개와 입장 표명\s*\n', '', text, flags=re.MULTILINE)
    text = re.sub(r'^자기소개와 입장 표명\s*$', '', text, flags=re.MULTILINE)
    # 결론 이후 메타 코멘트 제거
    cm = re.search(r'^### 결론', text, re.MULTILINE)
    if cm:
        after = text[cm.start():]
        after = re.sub(r'\n\s*\*?참고[\s:：].*', '', after, flags=re.DOTALL)
        after = re.sub(r'\n\s*\*?주[\s:：].*', '', after, flags=re.DOTALL)
        text = text[:cm.start()] + after
    # 끊김 패턴 수리: "을합니다" → "을 합니다", 조사+동사 바로 붙은 경우
    text = re.sub(r'([을를이가은는에])합니다', r'\1 합니다', text)
    text = re.sub(r'([을를이가은는에])하[게면고]', lambda m: m.group(1) + ' 하' + m.group(0)[-1], text)
    # 볼드 안 영어 전용 텍스트 제거 (**According to...** 등 검색 결과 제목 유출)
    text = re.sub(r'\*{2}[a-zA-Z][a-zA-Z\s,.\-;:\'\"()]{5,}\*{2}', '', text)
    # 깨진 혼합어 제거 ("카티rophic", "머천cies" 등)
    # 한국어 2글자+영어 3글자 이상 = 깨진 단어 ("카티rophic")
    text = re.sub(r'[가-힣]{2,}[a-zA-Z]{3,}[가-힣]*', '', text)
    # 영어 3글자+한국어 3���자 이상 = 깨진 단어 ("techno단지에서")
    # 한국어 1~2글자(조사: 의/에/는/가/를 등)는 유지 → "McKinsey의" 보존
    text = re.sub(r'[a-zA-Z]{3,}[가-힣]{3,}[a-zA-Z]*', '', text)
    # 불완전 문장 정리: 마지막 문장이 끝맺음 없이 끊겼으면 제거
    lines = text.rstrip().split('\n')
    if lines:
        last = lines[-1].rstrip()
        # 소제목이 아닌 일반 문장이 마침표/물음표/느낌표/볼드 없이 끝나면 불완전
        if last and not last.startswith('###') and not re.search(r'[.?!다까요\*]$', last):
            lines = lines[:-1]
            text = '\n'.join(lines)
    return text


def _has_cot_leakage(text: str) -> bool:
    """영어 CoT 유출 감지."""
    cot_patterns = [
        r'\b(?:First|Second|Third|Next|Then|Finally),?\s+I\b',
        r'\bI (?:need|should|will|can|must)\b',
        r'\bLet me\b',
        r'\bIn order to\b',
        r'\bthe (?:answer|response|argument|topic)\b',
        r'\b(?:Okay|OK),?\s+so\b',
        r'\bHmm\b',
        r'\bAssuming\b',
    ]
    for pattern in cot_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            return True

    # 영어 비율이 30% 초과하면 CoT 유출로 판단
    korean_chars = len(re.findall(r'[가-힣]', text))
    english_chars = len(re.findall(r'[a-zA-Z]', text))
    if korean_chars + english_chars > 0:
        if english_chars / (korean_chars + english_chars) > 0.3:
            return True

    return False


def _is_valid_speech(speech: str) -> bool:
    """최소 검증: 20자 이상, CoT 유출 없음, 외국어 깨짐 없음."""
    if not speech or len(speech.strip()) < 20:
        return False
    # 중국어/일본어 깨진 문자 감지
    if re.search(r'[\u4e00-\u9fff。，]', speech):
        return False
    # 한국어 비율이 너무 낮으면 무효
    korean_chars = len(re.findall(r'[가-힣]', speech))
    if korean_chars < 10:
        return False
    if _has_cot_leakage(speech):
        return False
    return True


# ── XML 도구 호출 폴백 파서 ──────────────────────────────────────────────────

def _parse_xml_tool_calls(content: str) -> List[Dict]:
    """<tool_call> XML 블록을 파싱."""
    tool_calls = []
    for block in re.findall(r'<tool_call>(.*?)</tool_call>', content, re.DOTALL):
        func_match = re.search(r'<function=(\w+)>(.*?)</function>', block, re.DOTALL)
        if not func_match:
            continue
        func_name = func_match.group(1)
        args: Dict[str, str] = {}
        for param in re.finditer(r'<parameter=(\w+)>\s*(.*?)\s*</parameter>', func_match.group(2), re.DOTALL):
            args[param.group(1)] = param.group(2).strip()
        tool_calls.append({"name": func_name, "args": args, "id": f"call_{func_name}_{uuid.uuid4().hex[:6]}"})
    return tool_calls


# ── 입론 프롬프트 ────────────────────────────────────────────────────────────

_MAX_TOOL_ROUNDS = 3
_MAX_TOOL_RESULT_CHARS = 800


def _truncate_tool_result(result: str, max_chars: int = _MAX_TOOL_RESULT_CHARS) -> str:
    if len(result) <= max_chars:
        return result
    return result[:max_chars] + "\n[일부만 표시]"


# 사전 생성된 검색 쿼리 로드
_SEARCH_QUERIES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "search_queries.json")
_SEARCH_QUERIES: Dict = {}
if os.path.exists(_SEARCH_QUERIES_PATH):
    with open(_SEARCH_QUERIES_PATH, encoding="utf-8") as _f:
        _SEARCH_QUERIES = json.load(_f)

# 에이전트별 쿼리 인덱스 (같은 stance 에이전트가 다른 쿼리를 사용하도록)
_query_idx: Dict[str, int] = {}


def _pre_search(topic: str, stance: str, topic_id: str = "") -> Tuple[str, List[Dict]]:
    """입론 전 사전 검색. search_queries.json 쿼리만 사용.

    Returns:
        (검색 결과 텍스트, tool_calls_log)
    """
    tool_calls_log: List[Dict] = []
    results = []

    # search_queries.json에서 쿼리 가져오기 (에이전트마다 다른 쿼리 순환)
    query = ""
    if topic_id and topic_id in _SEARCH_QUERIES:
        keywords = _SEARCH_QUERIES[topic_id].get(stance, [])
        if keywords:
            key = f"{topic_id}_{stance}"
            idx = _query_idx.get(key, 0)
            query = keywords[idx % len(keywords)]
            _query_idx[key] = idx + 1

    if not query:
        # fallback: 토픽 핵심어 + stance
        topic_short = topic.split("아닌")[0].strip() if "아닌" in topic else topic[:20]
        stance_kr = "찬성 근거 통계" if stance == "PRO" else "반대 근거 문제점 통계"
        query = f"{topic_short} {stance_kr}"

    tool_calls_log.append({"name": "search_web", "args": {"query": query}})
    web_result = search_web.invoke({"query": query})
    results.append(_truncate_tool_result(web_result))

    return "\n\n".join(results), tool_calls_log


def _build_opening_prompt(
    topic: str, stance: str, agent_name: str,
    search_results: str,
) -> str:
    stance_kr = "찬성" if stance == "PRO" else "반대"

    return f"""아래 참고 자료를 바탕으로 '{topic}'에 대한 {stance_kr} 입론을 작성하라.

[참고 자료]
{search_results}

[구조]
- "{agent_name}"이라고 자기소개
- 논거 2개, 각 3~5줄. 구체적 사례·데이터·국가 비교를 반드시 포함하라
- 핵심 문장에 **강조** 사용
- 일반론 금지. "~은 문제입니다" 수준의 막연한 주장 대신, 구체적 사례와 수치를 들어 설득하라

[인용 규칙 — 가장 중요]
- 반드시 위 [참고 자료]에서 기관명·수치·사례를 직접 인용하라
- 참고 자료에 없는 수치나 연구를 절대 지어내지 마라
- "~에 따르면"으로 인용할 때 반드시 참고 자료에 나온 출처명을 그대로 사용하라
- 확실하지 않으면 수치 없이 논리로 주장하라

[형식]
- 반드시 합니다체(격식체). 모든 문장을 "~합니다", "~입니다", "~됩니다"로 끝내라
- 한국어로 작성. 고유명사만 영어 허용

반드시 아래 형식으로만 출력:

### 답변 시작
### 자기소개와 입장 표명
(자기소개와 입장)
### 논거 1: 소제목
(논거)
### 논거 2: 소제목
(논거)
### 결론
(결론)
### 답변 끝"""


# ── 입론 생성 (사전 검색 + 단일 LLM 호출) ───────────────────────────────────

def _generate_opening(agent: Dict, prompt: str) -> Tuple[str, str]:
    """단일 LLM 호출로 입론 생성. 1회 재시도 + fallback."""
    messages = [
        SystemMessage(content=agent["system_prompt"]),
        HumanMessage(content=prompt),
    ]

    response: AIMessage = _invoke_with_retry(_llm, messages, label="opening")
    raw = response.content if isinstance(response.content, str) else str(response.content)
    speech = _postprocess_speech(_extract_delimited_text(raw))

    if not _is_valid_speech(speech):
        logger.warning("[opening] speech 무효, 재시도")
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content='한국어로만 입론을 작성하세요.\n\n### 답변 시작\n(입론)\n### 답변 끝'))
        retry: AIMessage = _invoke_with_retry(_llm, messages, label="opening_retry")
        raw = retry.content if isinstance(retry.content, str) else str(retry.content)
        speech = _postprocess_speech(_extract_delimited_text(raw))

    return speech, raw


# ── 메인 노드 ─────────────────────────────────────────────────────────────────

def opening_arguments_node(state: DebateState) -> DebateState:
    """1단계 입론 노드."""
    global _used_doc_ids
    _used_doc_ids = set()

    topic = state["topic"]
    history: List[DebateEntry] = list(state["debate_history"])
    agent_map = {a["agent_id"]: a for a in state["agents"]}

    _stance_counter: Dict[str, int] = {"PRO": 0, "CON": 0}
    _prior_queries: Dict[str, List[str]] = {"PRO": [], "CON": []}

    print(f"\n[1단계: 입론] 발언 순서: {state['speaking_order']}\n")

    for idx, speaker_id in enumerate(state["speaking_order"]):
        if speaker_id == "user":
            continue

        agent = agent_map[speaker_id]
        _stance_counter[agent["stance"]] += 1
        snum = _stance_counter[agent["stance"]]
        slabel = "찬성" if agent["stance"] == "PRO" else "반대"
        display = f"{slabel}{snum}"

        print(f"  [{display}] 입론 생성 중...")

        # 1. 사전 검색
        search_results, tool_calls_log = _pre_search(
            topic, agent["stance"],
            topic_id=state.get("topic_id", ""),
        )

        # 2. 프롬프트 구성 + LLM 호출
        prompt = _build_opening_prompt(
            topic, agent["stance"], display, search_results,
        )
        final_text, raw = _generate_opening(agent, prompt)

        # 3. 자기소개 소제목 보장 (입론 전용)
        if '### 자기소개' not in final_text and '### 입장 표명' not in final_text:
            first_h = re.search(r'^### ', final_text, re.MULTILINE)
            if first_h and first_h.start() > 0:
                intro = final_text[:first_h.start()].strip()
                rest = final_text[first_h.start():]
                if intro:
                    final_text = f"### 자기소개와 입장 표명\n{intro}\n\n{rest}"
            elif not final_text.startswith('###'):
                final_text = f"### 자기소개와 입장 표명\n{final_text}"

        # 4. fallback
        if not _is_valid_speech(final_text):
            logger.warning("[opening] fallback 사용")
            final_text = (
                f"### 자기소개와 입장 표명\n"
                f"저는 {display}입니다. {topic}에 대해 {slabel} 입장입니다.\n\n"
                f"### 결론\n저는 {slabel} 입장을 유지합니다."
            )

        entry = DebateEntry(
            turn=idx, speaker_id=speaker_id, stance=agent["stance"],
            phase="opening", content=final_text, target_id=None,
            tool_calls_log=tool_calls_log, json_raw=raw,
        )
        history.append(entry)

        print(f"  [{display}] 입론 완료 (turn={entry['turn']})\n")

    history.sort(key=lambda e: e["turn"])
    print("[1단계: 입론] AI 에이전트 입론 완료 → 사용자 입론 대기\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": len(state["speaking_order"]),
        "current_speaker_index": len(state["speaking_order"]),
        "phase": "opening",
    })
