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
    api_key=os.environ.get("LLM_API_KEY", "fake"),
    temperature=0.4,  # 답변 분산 완화
    max_tokens=2560,  # plan + 자료 + 본문 합쳐 token 한도 도달 방지
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
    # 영어 단어 제거 제거함 — 고유명사/전문용어 오탐 방지 (프롬프트로 제어)
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
    # 평서체 → 격식체 rule-based 치환 (문장 끝 특정 패턴만)
    _FORMAL_MAP = [
        # 긴 패턴부터 먼저 치환 (겹침 방지)
        (r'것이었다([\.,!?])', r'것이었습니다\1'),
        (r'것이다([\.,!?])', r'것입니다\1'),
        (r'할\s*수\s*있다([\.,!?])', r'할 수 있습니다\1'),
        (r'할\s*수\s*없다([\.,!?])', r'할 수 없습니다\1'),
        (r'있었다([\.,!?])', r'있었습니다\1'),
        (r'없었다([\.,!?])', r'없었습니다\1'),
        (r'이었다([\.,!?])', r'이었습니다\1'),
        (r'하였다([\.,!?])', r'하였습니다\1'),
        (r'되었다([\.,!?])', r'되었습니다\1'),
        # 기본 패턴
        (r'([^가-힣ㅏ-ㅣ])이다([\.,!?])', r'\1입니다\2'),
        (r'한다([\.,!?])', r'합니다\1'),
        (r'된다([\.,!?])', r'됩니다\1'),
        (r'있다([\.,!?])', r'있습니다\1'),
        (r'없다([\.,!?])', r'없습니다\1'),
        (r'했다([\.,!?])', r'했습니다\1'),
        (r'됐다([\.,!?])', r'됐습니다\1'),
        (r'였다([\.,!?])', r'였습니다\1'),
        (r'봤다([\.,!?])', r'봤습니다\1'),
        (r'왔다([\.,!?])', r'왔습니다\1'),
        (r'갔다([\.,!?])', r'갔습니다\1'),
    ]
    for pat, rep in _FORMAL_MAP:
        text = re.sub(pat, rep, text)
    # 소제목 마커 정규화 — 다양한 변형을 모두 "### "로 통일
    # (1) 줄 앞 공백 + # 변형 정규화 (leading whitespace 허용, # 개수·간격·반복 변형 모두)
    text = re.sub(
        r'^[ \t]*(?:#+[ \t]*)+(?=[가-힣A-Za-z0-9])',
        '### ',
        text,
        flags=re.MULTILINE,
    )
    # (2) 볼드로 감싼 헤딩 (**### 논거 1**) 정리 → ### 논거 1
    text = re.sub(
        r'^\s*\*{1,3}\s*(###\s*[^*\n]+?)\s*\*{1,3}\s*$',
        r'\1',
        text,
        flags=re.MULTILINE,
    )
    # (3) # 마커 완전 누락 — "자기소개/논거 N/결론/입장 표명" 으로 시작하는 줄에 ### 자동 추가
    _HEADING_RE = re.compile(
        r'^\s*(자기소개[가-힣\s]*?|입장\s*표명|논거\s*\[?\s*\d+\s*\]?|결론)\s*(?:[:：]|$)',
    )
    _fixed_lines = []
    for _ln in text.split('\n'):
        _s = _ln.lstrip()
        if not _s.startswith('###') and _HEADING_RE.match(_ln):
            _fixed_lines.append('### ' + _s)
        else:
            _fixed_lines.append(_ln)
    text = '\n'.join(_fixed_lines)

    # 연속 중복 줄/문장 제거 (모델이 같은 내용을 두 번 찍는 경우)
    _lines = text.split('\n')
    _dedup = []
    _prev_norm = ""
    for _ln in _lines:
        _norm = re.sub(r'\s+', ' ', _ln).strip()
        # 짧은 줄은 정상 허용, 10자 이상 내용 줄이 직전과 80% 이상 유사하면 스킵
        if _norm and len(_norm) >= 10 and _prev_norm and len(_prev_norm) >= 10:
            # 직전 줄이 이번 줄의 prefix 이거나 그 반대면 중복 간주
            if _norm.startswith(_prev_norm[:30]) or _prev_norm.startswith(_norm[:30]):
                continue
        _dedup.append(_ln)
        if _norm:
            _prev_norm = _norm
    text = '\n'.join(_dedup)
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
    # 영어 3글자+한국어 3글자 이상 = 깨진 단어 ("techno단지에서")
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


def validate_quality(speech: str, min_chars: int = 20) -> Tuple[bool, str]:
    """품질보증 검증. (통과 여부, 실패 사유) 반환.

    모든 스테이지에서 공통으로 사용하는 통합 검증 함수.
    """
    if not speech or len(speech.strip()) < min_chars:
        return False, f"길이 부족 ({len(speech.strip()) if speech else 0}자)"

    # 1. 깨진 문자: 중국어/일본어/중국어 문장부호
    if re.search(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uff60。，]', speech):
        return False, "외국어 깨진 문자"

    # 2. 한국어 비율
    korean = len(re.findall(r'[가-힣]', speech))
    english = len(re.findall(r'[a-zA-Z]', speech))
    if korean < 10:
        return False, f"한국어 부족 ({korean}자)"
    if korean + english > 0 and english / (korean + english) > 0.5:
        return False, f"영어 비율 과다 ({english / (korean + english):.0%})"

    # 3. CoT 유출 (영어 사고 과정)
    cot_patterns = [
        r'\b(?:First|Second|Third|Next|Then|Finally),?\s+I\b',
        r'\bI (?:need|should|will|can|must)\b',
        r'\bLet me\b', r'\bIn order to\b',
        r'\b(?:Okay|OK),?\s+so\b', r'\bHmm\b',
    ]
    for p in cot_patterns:
        if re.search(p, speech, re.IGNORECASE):
            return False, "CoT 유출"

    # 4. 문장 끊김: 마지막 줄이 끝맺음 없이 끊긴 경우
    lines = speech.rstrip().split('\n')
    last = lines[-1].strip() if lines else ""
    if last and not last.startswith('###') and len(last) > 10:
        if not re.search(r'[.?!다까요)\*"]$', last):
            return False, f"문장 끊김"

    # 5. 빈 소제목: ### 뒤에 내용 없이 바로 ###이 오거나, ### 로 끝나는 경우
    if re.search(r'###[^\n]*\n\s*###', speech):
        return False, "빈 소제목"
    if re.search(r'###\s*$', speech.rstrip()):
        return False, "빈 소제목으로 끝남"

    # 6. 깨진 숫자/콤마 잔해 (후처리 후 잔여)
    if re.search(r'(?:^|\n)\s*[\d,\s]{5,}\s*(?:$|\n)', speech):
        return False, "깨진 숫자 잔해"

    # 7. 동일 문장 반복 (30자 이상 문장이 2회 출현)
    sentences = [s.strip() for s in re.split(r'[.!?]\s+', speech) if len(s.strip()) > 30]
    if len(sentences) != len(set(sentences)):
        return False, "문장 반복"

    # 8. 격식체 검증은 _postprocess_speech의 rule-based 치환으로 대체함 (탈락 없이 자동 교정)

    return True, "OK"


# ── 스테이지별 소제목 검증 ──────────────────────────────────────────────────

_REQUIRED_HEADINGS = {
    "opening": ["자기소개", "논거 1", "논거 2", "결론"],
    "role_reversal": ["논거 1", "논거 2", "결론"],
}


def check_headings(speech: str, stage: str) -> Tuple[bool, List[str]]:
    """스테이지별 필수 소제목(### 포함) 존재 여부 확인. (통과, 누락 목록) 반환."""
    required = _REQUIRED_HEADINGS.get(stage, [])
    if not required:
        return True, []
    missing = [h for h in required if not re.search(rf'###\s*{re.escape(h)}', speech)]
    return len(missing) == 0, missing


def _is_valid_speech(speech: str) -> bool:
    """하위 호환용 래퍼. 입론/역할반전용 (min_chars=20)."""
    ok, reason = validate_quality(speech, min_chars=20)
    if not ok:
        logger.warning("[품질검증] 실패: %s", reason)
    return ok


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

def _get_focus_area(stance: str, topic_id: str = "", index: int = 0) -> str:
    """에이전트별 논증 초점 영역을 반환한다.

    `data/search_queries.json` 의 (topic_id, stance) 키워드 리스트에서 명시된
    `index` 위치의 키워드를 반환한다. 이전에는 모듈 전역 카운터를 썼는데
    프로세스 재시작 사이에만 0으로 리셋되어 세션마다 다른 인덱스에서 시작하는
    문제가 있었음 — 이제 호출자가 명시적 인덱스를 넘긴다.

    Parameters
    ----------
    index : 같은 진영 내 몇 번째 AI 인지 (0-based).
        같은 stance 다른 AI 끼리 다른 focus 를 받도록 호출자가 위치 계산해서 전달.

    Returns
    -------
    str : focus_area 키워드. topic_id/진영 미매칭 시 "".
    """
    if topic_id and topic_id in _SEARCH_QUERIES:
        keywords = _SEARCH_QUERIES[topic_id].get(stance, [])
        if keywords:
            return keywords[index % len(keywords)]
    return ""


def _focus_index_within_stance(agent: Dict, agents: list) -> int:
    """agents 리스트에서 같은 stance 의 몇 번째 위치인지 (0-based) 반환."""
    same = [a for a in agents if a.get("stance") == agent.get("stance")]
    for i, a in enumerate(same):
        if a.get("agent_id") == agent.get("agent_id"):
            return i
    return 0


def _pre_search(topic: str, stance: str, topic_id: str = "") -> Tuple[str, List[Dict]]:
    """역할반전 등 다른 모듈 호환용 사전 검색. focus area 기반 단일 검색."""
    tool_calls_log: List[Dict] = []
    query = _get_focus_area(stance, topic_id)
    if not query:
        topic_short = topic.split("아닌")[0].strip() if "아닌" in topic else topic[:20]
        stance_kr = "찬성 근거 통계" if stance == "PRO" else "반대 근거 문제점 통계"
        query = f"{topic_short} {stance_kr}"

    from src.graph.llm import safe_search_invoke
    web_result = safe_search_invoke({"query": query})
    result = _truncate_tool_result(web_result)
    tool_calls_log.append({"name": "search_web", "args": {"query": query}, "result": result})
    return result, tool_calls_log


def _build_plan_prompt(topic: str, stance: str, focus_area: str) -> str:
    """Step 1 (Plan) — focus_area 기반 논거 outline + 검색 쿼리 도출."""
    stance_kr = "찬성" if stance == "PRO" else "반대"
    return f"""[Step 1 — 계획 수립] '{topic}'에 대한 {stance_kr} 입론 작성 전 계획을 세워라.

[너의 논증 초점]
focus_area: {focus_area}

[지시]
focus_area 의 두 측면을 어떻게 자기 진영 옹호 논거로 풀어낼지 계획하고, 각 논거를 뒷받침할 검색 쿼리를 도출하라.

[출력 형식 — JSON 객체 하나만]
```json
{{
  "argument_outline": [
    "논거 1 핵심 주장 (1문장, focus_area 의 한 측면, {stance_kr} 진영 옹호 방향)",
    "논거 2 핵심 주장 (1문장, focus_area 의 다른 측면, {stance_kr} 진영 옹호 방향)"
  ],
  "search_queries": [
    "논거 1 자료 검색용 쿼리 (focus_area 핵심 키워드 포함, {stance_kr} 측 옹호 자료가 hit 될 방향)",
    "논거 2 자료 검색용 쿼리"
  ]
}}
```

[규칙]
- 쿼리는 focus_area 의 핵심 키워드를 그대로 포함하라. 임의 변형 금지.
- 쿼리는 자기 진영 ({stance_kr}) 의 주장을 뒷받침할 자료가 hit 되는 방향으로 짜라 (반대 진영을 비판하는 자료 X, 자기 진영을 옹호하는 자료 O).
- 입론 본문은 절대 작성하지 마라. 위 JSON 만 출력하라."""


def _extract_plan_json(raw: str) -> dict:
    """plan LLM 출력에서 JSON 추출. 실패 시 빈 plan 반환."""
    text = (raw or "").strip()
    # ```json ... ``` 추출
    m = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # 첫 { ... } 추출
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch != '{':
            continue
        try:
            parsed, _ = decoder.raw_decode(text[idx:])
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return {"argument_outline": [], "search_queries": []}


def _generate_plan(agent: Dict, topic: str, stance: str, focus_area: str) -> Tuple[dict, str]:
    """Step 1 실행 — Plan LLM 호출."""
    plan_prompt = _build_plan_prompt(topic, stance, focus_area)
    messages = [
        SystemMessage(content=agent["system_prompt"]),
        HumanMessage(content=plan_prompt),
    ]
    response = _invoke_with_retry(_llm, messages, label="opening_plan")
    raw = response.content if isinstance(response.content, str) else str(response.content)
    plan = _extract_plan_json(raw)
    return plan, raw


def _build_opening_prompt(
    topic: str, stance: str, agent_name: str,
    focus_area: str = "",
    plan: Optional[dict] = None,
    search_results: Optional[List[dict]] = None,
) -> str:
    stance_kr = "찬성" if stance == "PRO" else "반대"

    focus_block = ""
    if focus_area:
        focus_block = f"\n[너의 논증 초점]\nfocus_area: {focus_area}\n"

    plan_block = ""
    if plan and plan.get("argument_outline"):
        plan_block = "\n[Step 1 — 너의 계획 (이 방향으로 논거 전개)]\n"
        for i, outline in enumerate(plan["argument_outline"], 1):
            plan_block += f"논거 {i}: {outline}\n"

    context_block = ""
    if search_results:
        context_block = "\n[Step 2 — 검색 결과]\n"
        for i, sr in enumerate(search_results, 1):
            context_block += f"\n--- 쿼리 {i}: {sr.get('query','')} ---\n{sr.get('result','')}\n"

    return f"""[Step 3 — 입론 작성] '{topic}'에 대한 {stance_kr} 입론을 작성하라.
{focus_block}{plan_block}{context_block}
[구조]
- "{agent_name}"이라고 자기소개
- 논거 2개, 각 3~5줄. Step 1 의 계획에 따라 작성하라.
- **각 논거 본문에 핵심 문장 1개를 반드시 `**굵은 글씨**` 로 감싸 강조한다.** 핵심 사실·통계·결론 문장 1개를 골라 `**...**` 마크다운 볼드로 표시. 논거당 최소 1회, 최대 2회.
- 막연한 주장 금지

[자료 활용 룰 — 절대 준수]
- 검색 결과의 자료가 **자기 진영·focus_area 의 주장을 직접 옹호**할 때만 인용하라.
- 자료가 자기 주장과 충돌하거나 (자기 진영을 비판하는 자료), 주제·focus_area 와 무관하면 **인용하지 마라**. 그 경우 사례·수치 없이 **순수 논리·일반화 표현으로 논거를 전개**하라 ("상당수", "대체로", "최근" 등 허용).
- 자료를 자기 framing 에 맞춰 곡해하지 마라. 자료의 원 결론·tone 과 다른 해석으로 끌어가지 마라.
- 머릿속 수치·기관명·법안명은 절대 금지. 검색 결과에 명시된 것만 인용.
- 존재하지 않는 연구·기관·법안·사건을 지어내지 마라.

[형식]
- 반드시 합니다체(격식체). 모든 문장을 "~합니다", "~입니다", "~됩니다"로 끝내라
- 한국어로 작성. 고유명사만 영어 허용

반드시 아래 형식으로만 출력:

### 답변 시작
### 자기소개와 입장 표명
(자기소개와 입장)
### 논거 1: 소제목
(논거 본문 — 핵심 문장 1개는 반드시 `**굵은 글씨**` 로 강조)
### 논거 2: 소제목
(논거 본문 — 핵심 문장 1개는 반드시 `**굵은 글씨**` 로 강조)
### 결론
(결론)
### 답변 끝"""


# ── 인용-검색 일치 검증 (구조적 강제) ────────────────────────────────────────

_CITATION_PATTERNS = [
    # 수치·%·금액 (연도·세기 등 일반 숫자는 제외)
    re.compile(r'\d+(?:\.\d+)?\s*%'),                        # 3.5%, 20%
    re.compile(r'\d+(?:,\d+)*\s*(?:조|억|만)\s*(?:달러|원)?'),  # 39조 달러
    re.compile(r'\$\s*\d+(?:,\d+)*(?:\.\d+)?\s*(?:조|억|만|억만|trillion|billion|million)?'),
    re.compile(r'\d+(?:\.\d+)?\s*(?:p|%p|%포인트|퍼센트\s*포인트)'),  # 0.5%p
    re.compile(r'\d+(?:\.\d+)?\s*(?:배|명|만명|억명|대)'),     # 300만명, 2배
    # 기관명·보고서 (한국어 고유명)
    re.compile(r'(?:옥스포드\s*이코노믹스|IMF|OECD|EBRD|IHS\s*Markit|World\s*Bank|WB|IEA|UN)'),
    re.compile(r'(?:국회예산정책처|국가안보전략연구원|한국은행|KDI|KOTRA)'),
    # 연도+구체 사건 (예: "2020년 솔레이마니 사건")
    re.compile(r'\b(?:19|20)\d{2}\s*년\s*(?:[가-힣A-Za-z0-9]+(?:\s*사건|\s*공격|\s*제재|\s*사태|\s*협정|\s*조약|\s*회담))'),
]


def _has_specific_citation(text: str) -> List[str]:
    """구체적 수치·기관명·연도사건 인용을 추출한다. 검증 필요 인용이 있을 때만 리스트 반환."""
    found: List[str] = []
    for pat in _CITATION_PATTERNS:
        for m in pat.findall(text):
            s = m if isinstance(m, str) else " ".join(m)
            if s and s not in found:
                found.append(s)
    return found


def _validate_citation_search(speech: str, tool_calls_log: List[Dict]) -> Tuple[bool, str]:
    """구체 인용이 있는데 search_web 호출이 0이면 실패. 재생성 힌트 반환.

    Returns:
        (ok, feedback_msg)
        - ok=True: 통과 (인용 없음 or 인용 있고 검색도 있음)
        - ok=False: 실패 — feedback_msg를 재생성 프롬프트로 전달
    """
    citations = _has_specific_citation(speech)
    searches = [tc for tc in (tool_calls_log or []) if tc.get("name") == "search_web"]
    if citations and not searches:
        cited_examples = ", ".join(citations[:4])
        msg = (
            f"방금 발언에 구체적 수치/기관명이 포함되어 있는데 search_web을 호출하지 않았습니다. "
            f"(감지된 인용: {cited_examples})\n"
            f"두 가지 선택 중 하나를 반드시 하세요:\n"
            f"  (A) search_web을 **지금 호출**해서 그 수치·기관을 검증한 뒤, 검색 결과에 나온 내용만 인용해서 재작성하세요.\n"
            f"  (B) 구체 수치·기관명을 모두 빼고 '상당수', '대체로', '최근', '많은 경우' 같은 **일반화 표현**으로 논리 중심 발언을 재작성하세요.\n"
            f"반드시 전체 발언을 다시 작성하세요."
        )
        return False, msg
    return True, ""


# ── 입론 생성 (사전 검색 + 단일 LLM 호출) ───────────────────────────────────

def _generate_opening(
    agent: Dict, topic: str, stance: str, agent_name: str, focus_area: str,
) -> Tuple[str, str, List[Dict]]:
    """Plan-and-Execute 입론 생성: Plan → Search → Generate.

    1) Plan: focus_area 기반 논거 outline + 검색 쿼리 도출 (LLM 호출)
    2) Search: plan 의 쿼리로 search_web (병렬 가능)
    3) Generate: plan + 검색 결과 → 본문 작성 (tools 비활성, 1회 호출)

    ReAct 의 "사후 보강 검색" 패턴이 자료 misread 를 유발해 plan-and-execute 로 전환.
    Pre-Act (2025) 와 RAG hallucination 연구 참조.
    """
    try:
        from src.graph.vector_store import set_current_stance
        set_current_stance(agent.get("stance"))
    except Exception:
        pass

    _tag = f"agent={agent.get('agent_id','?')} stance={agent.get('stance','?')}"
    _t0 = time.time()
    def _tlog(step: str, dt: float, extra: str = ""):
        msg = f"[opening_timing] t={time.time()-_t0:6.2f}s {_tag} step={step} dt={dt:.2f}s {extra}"
        print(msg, flush=True)
        logger.warning(msg)

    tool_calls_log: List[Dict] = []

    # ── Step 1: Plan ──────────────────────────────────────────────────────
    _s = time.time()
    plan, plan_raw = _generate_plan(agent, topic, stance, focus_area)
    queries = plan.get("search_queries", []) or []
    # Fallback chain — plan JSON 추출 실패 또는 빈 queries 일 때
    if not queries:
        stance_kr = "찬성" if stance == "PRO" else "반대"
        if focus_area:
            queries = [focus_area]  # primary fallback
        else:
            # 마지막 fallback — topic + stance 키워드
            topic_short = topic.split("아닌")[0].strip() if "아닌" in topic else topic[:30]
            queries = [f"{topic_short} {stance_kr} 근거"]
        logger.warning("[opening] plan queries empty, fallback to %r", queries)
    _tlog("plan", time.time() - _s, f"queries={len(queries)} outline={len(plan.get('argument_outline', []))}")

    # ── Step 2: Search ────────────────────────────────────────────────────
    search_results: List[Dict] = []
    from src.graph.llm import safe_search_invoke
    for q in queries[:3]:  # 최대 3 쿼리
        _s = time.time()
        result = safe_search_invoke({"query": q})
        result = _truncate_tool_result(str(result))
        _tlog("search", time.time() - _s, f"query={q!r}")
        tool_calls_log.append({"name": "search_web", "args": {"query": q}, "result": result})
        search_results.append({"query": q, "result": result})

    # ── Step 3: Generate ──────────────────────────────────────────────────
    gen_prompt = _build_opening_prompt(
        topic, stance, agent_name,
        focus_area=focus_area, plan=plan, search_results=search_results,
    )
    messages = [
        SystemMessage(content=agent["system_prompt"]),
        HumanMessage(content=gen_prompt),
    ]
    _s = time.time()
    response = _invoke_with_retry(_llm, messages, label="opening_generate")
    _tlog("generate", time.time() - _s)

    raw = response.content if isinstance(response.content, str) else str(response.content)
    speech = _postprocess_speech(_extract_delimited_text(raw))

    # 품질 검증 + 소제목 검증 → 실패 시 1회 재시도
    ok, reason = validate_quality(speech, min_chars=20)
    headings_ok, missing = check_headings(speech, "opening")

    if not ok or not headings_ok:
        retry_hint = ""
        if not ok:
            retry_hint += f"이전 응답이 부적절합니다 ({reason}). "
        if not headings_ok:
            retry_hint += f"다음 소제목이 빠져있습니다: {', '.join(missing)}. "
        retry_hint += "한국어로 반드시 모든 소제목을 포함하여 다시 작성하세요."
        logger.warning("[opening] 재시도: %s / 누락 소제목: %s", reason, missing)
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content=f'{retry_hint}\n\n### 답변 시작\n### 자기소개와 입장 표명\n(자기소개)\n### 논거 1: 소제목\n(논거)\n### 논거 2: 소제목\n(논거)\n### 결론\n(결론)\n### 답변 끝'))
        _s = time.time()
        retry: AIMessage = _invoke_with_retry(_llm, messages, label="opening_retry")
        _tlog("llm_call_quality_retry", time.time() - _s, f"reason={reason} missing={missing}")
        raw = retry.content if isinstance(retry.content, str) else str(retry.content)
        speech = _postprocess_speech(_extract_delimited_text(raw))

    # 구조적 강제: 구체 수치·기관명 인용이 있는데 search_web을 호출 안 했으면 재생성
    for attempt in range(2):
        cite_ok, cite_feedback = _validate_citation_search(speech, tool_calls_log)
        if cite_ok:
            break
        logger.warning("[opening] 인용-검색 불일치 감지, 재시도 %d/2", attempt + 1)
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content=cite_feedback))
        _s = time.time()
        retry: AIMessage = _invoke_with_retry(_llm_with_tools, messages, label="opening_cite_retry")
        _tlog(f"llm_call_cite_retry_{attempt+1}", time.time() - _s, f"emitted_tool_call={bool(getattr(retry,'tool_calls',None))}")
        # tool call 처리 (재시도 중에도 검색 가능)
        if hasattr(retry, "tool_calls") and retry.tool_calls:
            messages.append(retry)
            for tc in retry.tool_calls:
                tool_name = tc.get("name", "")
                tool_args = tc.get("args", {})
                tool_id = tc.get("id", "")
                if tool_name in _TOOL_MAP:
                    _ts = time.time()
                    result = _TOOL_MAP[tool_name].invoke(tool_args)
                    _tlog(f"cite_retry_tool_exec_{tool_name}", time.time() - _ts)
                    result = _truncate_tool_result(str(result))
                    tool_calls_log.append({"name": tool_name, "args": tool_args, "result": result})
                    messages.append(ToolMessage(content=result, tool_call_id=tool_id))
                    logger.info("[opening] retry tool call: %s(%s)", tool_name, tool_args)
            _s = time.time()
            retry = _invoke_with_retry(_llm, messages, label="opening_cite_retry_final")
            _tlog(f"llm_call_cite_retry_final_{attempt+1}", time.time() - _s)
        raw = retry.content if isinstance(retry.content, str) else str(retry.content)
        speech = _postprocess_speech(_extract_delimited_text(raw))

    _tlog("TOTAL", time.time() - _t0, f"output_chars={len(speech)} tool_calls={len(tool_calls_log)}")
    return speech, raw, tool_calls_log


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

        # 1. focus area — persona의 각도명을 우선 사용, 없으면 같은 stance 내 위치 기반
        focus_area = agent.get("focus_area") or _get_focus_area(
            agent["stance"],
            topic_id=state.get("topic_id", ""),
            index=_focus_index_within_stance(agent, state["agents"]),
        )
        if focus_area:
            print(f"    [focus] {focus_area}")

        # 캐시 lookup — hit 시 LLM 호출 건너뜀
        from src.cache.loader import load_opening as _cache_load_opening
        cached = _cache_load_opening(
            topic_id=state.get("topic_id", ""),
            stance=agent["stance"],
            intensity=agent.get("intensity", 3),
            focus_area=focus_area,
            variant_idx=state.get("cache_variant_idx"),
        )
        if cached is not None:
            print(f"  [{display}] [cache HIT] 입론")
            final_text, raw, tool_calls_log = cached, cached, []
        else:
            # Plan-and-Execute pipeline 으로 입론 생성
            #    Step 1 (Plan): focus_area → 논거 outline + 검색 쿼리
            #    Step 2 (Search): plan 쿼리로 search_web
            #    Step 3 (Generate): plan + 자료 → 본문 작성
            final_text, raw, tool_calls_log = _generate_opening(
                agent, topic, agent["stance"], display, focus_area,
            )

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
