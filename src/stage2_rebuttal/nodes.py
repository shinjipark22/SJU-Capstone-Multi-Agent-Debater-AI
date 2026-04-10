"""
nodes.py — 2단계: 연쇄 논박(Chained Rebuttal) 노드

[설계 노트]
    - delimiter 기반 자연어 출력
    - 단일 LLM 호출 (max_tokens=256)
    - DeepSeek-R1-Distill-Qwen-14B 최적화
"""

from __future__ import annotations

import logging
import os
import random
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
    _remove_english_blocks,
    search_web,
    search_vector_db,
    _LLM_KWARGS,
)


def _is_valid_rebuttal(speech: str) -> bool:
    """연쇄논박 전용 검증. 입론보다 영어 임계값 완화 (짧은 텍스트 특성 반영)."""
    if not speech or len(speech.strip()) < 15:
        logger.warning("[rebuttal 검증] 실패: 15자 미만 (%d자)", len(speech.strip()) if speech else 0)
        return False
    # 영어 CoT 패턴 감지
    cot_patterns = [
        r'\b(?:First|Second|Third|Next|Then|Finally),?\s+I\b',
        r'\bI (?:need|should|will|can|must)\b',
        r'\bLet me\b',
        r'\bIn order to\b',
        r'\b(?:Okay|OK),?\s+so\b',
        r'\bHmm\b',
        r'\bAssuming\b',
    ]
    for pattern in cot_patterns:
        m = re.search(pattern, speech, re.IGNORECASE)
        if m:
            logger.warning("[rebuttal 검증] 실패: CoT 패턴 '%s'", m.group())
            return False
    # 영어 비율 50% 초과 시 유출
    korean_chars = len(re.findall(r'[가-힣]', speech))
    english_chars = len(re.findall(r'[a-zA-Z]', speech))
    if korean_chars + english_chars > 0:
        ratio = english_chars / (korean_chars + english_chars)
        if ratio > 0.5:
            logger.warning("[rebuttal 검증] 실패: 영어 비율 %.1f%%", ratio * 100)
            return False
    return True
from src.state import (
    DebateEntry,
    DebateState,
    build_chained_rebuttal_pairs,
)

# ── 연쇄논박 전용 LLM (DeepSeek — 반박 생성용) ─────────────────────────────
_rebuttal_llm = ChatOpenAI(**{**_LLM_KWARGS, "max_tokens": 1024})

# ── 분석 모델 (Qwen2.5-7B — 검색 판단 + 약점 분석 + 입장 검증, GPU 1) ────────
_QWEN_BASE_URL = os.environ.get("QWEN_BASE_URL", "http://localhost:8001/v1")
_qwen_llm = ChatOpenAI(
    model="Qwen/Qwen2.5-7B-Instruct",
    base_url=_QWEN_BASE_URL,
    api_key="fake",
    temperature=0.3,
    max_tokens=200,
    timeout=30,
)


def _decide_search(target_argument: str, attack_style: str) -> str:
    """Qwen 7B가 검색 필요 여부를 판단한다. 불필요 시 빈 문자열."""
    try:
        messages = [
            HumanMessage(content=f"""다음 주장을 반박하려 한다. 반박에 통계나 사실 확인이 필요하면 검색 키워드를 한국어 30자 이내로 출력하라.
논리만으로 반박 가능하면 "불필요"라고만 출력하라.

주장: {target_argument[:200]}""")
        ]
        response = _qwen_llm.invoke(messages)
        result = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()

        if "불필요" in result or len(result) < 3:
            logger.info("[qwen7b] 검색 불필요")
            return ""

        # 검색 키워드 추출 (첫 줄만)
        query = result.split('\n')[0].strip().strip('"').strip("'")
        logger.info("[qwen7b] 검색 키워드: '%s'", query[:40])
        return query[:40]
    except Exception as e:
        logger.warning("[qwen7b] _decide_search 오류: %s", e)
        return ""


# ── 텍스트 추출 (delimiter 없이, <think> + 영어 제거 후 한국어만) ────────────

def _check_stance(text: str, expected_stance: str, topic: str) -> bool:
    """Qwen 7B로 발언이 기대 입장과 일치하는지 판별한다. 일치하면 True."""
    stance_kr = "찬성" if expected_stance == "PRO" else "반대"
    try:
        messages = [
            HumanMessage(content=f"""주제: {topic[:100]}

발언: {text[:200]}

이 발언은 위 주제에 대해 "찬성"인가 "반대"인가? 한 단어로만 답하라.""")
        ]
        response = _qwen_llm.invoke(messages)
        result = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
        first_word = result.split()[0] if result.split() else ""
        detected = "찬성" if "찬성" in first_word else ("반대" if "반대" in first_word else "")
        if detected and detected != stance_kr:
            logger.warning("[stance_check] 입장 혼동: 기대=%s, 감지=%s", stance_kr, detected)
            return False
        return True
    except Exception as e:
        logger.warning("[stance_check] 오류: %s", e)
        return True


def _generate_attack_question(attack_text: str, stance: str, topic: str) -> str:
    """Qwen 7B로 공격 발언의 흐름에 맞는 마무리 질문을 생성한다."""
    try:
        messages = [
            HumanMessage(content=f"""다음 공격 발언을 읽고, 이 흐름에 맞는 마무리 질문을 1개 만들어라.

공격 발언: {attack_text[:300]}

규칙:
- 공격 내용과 자연스럽게 이어지는 질문
- "~할 수 있습니까?", "~라고 보십니까?", "~지 않습니까?" 형태
- 반드시 한국어로만 출력. 영어/중국어 등 다른 언어 금지
- 합니다체
- 한 문장만 출력. ?로 끝나야 함
- 설명하지 말고 질문만 출력

예시: 그렇다면 상대는 이 문제를 어떻게 해결할 수 있다고 보십니까?""")
        ]
        response = _qwen_llm.invoke(messages)
        result = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
        for line in result.split('\n'):
            line = line.strip()
            if line.endswith('?') and re.search(r'[가-힣]', line):
                return line
        return ""
    except Exception as e:
        logger.warning("[qwen7b] _generate_attack_question 오류: %s", e)
        return ""


def analyze_weakness(target_speech: str, topic: str) -> str:
    """Qwen 7B로 상대 논거의 핵심 약점을 분석한다. DeepSeek 반박 생성 전에 호출."""
    try:
        messages = [
            HumanMessage(content=f"""다음 주장의 가장 약한 부분을 1줄로 짚어라.

토론 주제: {topic[:80]}
상대 주장: {target_speech[:300]}

[필수] 반드시 한국어로만 답변하라. 영어, 중국어 등 다른 언어 사용 금지.
형식: "약점: (내용)" 한 줄만 출력.""")
        ]
        response = _qwen_llm.invoke(messages)
        result = response.content.strip() if isinstance(response.content, str) else str(response.content).strip()
        # 중국어/영어 유출 필터링
        if re.search(r'[\u4e00-\u9fff]', result):
            logger.warning("[qwen7b] 중국어 유출 감지, 결과 폐기")
            return ""
        # "약점:" 이후 추출
        if "약점:" in result:
            return result.split("약점:")[-1].strip()
        return result.split('\n')[0].strip()
    except Exception as e:
        logger.warning("[qwen7b] analyze_weakness 오류: %s", e)
        return ""


def _extract_rebuttal_text(content: str) -> str:
    """<think> 블록과 영어를 제거하고 한국어 문장만 추출한다."""
    text = content.strip()

    # 깨진 유니코드 제거
    text = text.replace('\ufffd', '')

    # <think> 제거
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
    text = text.replace('</think>', '').strip()

    # 영어 CoT 블록 제거 (입론과 동일 로직)
    text = _remove_english_blocks(text)

    # delimiter 추출: ### 반박 시작 ~ ### 반박 끝
    m = re.search(r'###\s*반박\s*시작\s*(?:###)?\s*\n?(.*?)\n?\s*###\s*반박\s*끝', text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()
    # '### 반박 시작' 이후 전체
    m = re.search(r'###\s*반박\s*시작\s*(?:###)?\s*\n?(.*)', text, re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip()

    # delimiter 없으면 기존 로직으로 fallback
    # 한국어가 포함된 줄만 추출
    korean_lines = []
    for l in text.split('\n'):
        s = l.strip()
        if not s or not re.search(r'[가-힣]', s):
            continue
        # 메타 문장 제거
        if s.startswith('상대의 주장을 반박') or s.startswith('반박'):
            if len(s) < 15:
                continue
        # 번호 매김 제거
        s = re.sub(r'^\d+\.\s*', '', s)
        s = re.sub(r'\s+\d+\.\s+', ' ', s)
        s = re.sub(r'^\d+\.\s*\d+\.\s*', '', s)
        # 빈 볼드/다중 공백 정리
        s = re.sub(r'\*{2,}\s*\*{2,}', '', s)
        s = re.sub(r'\s{2,}', ' ', s).strip()
        if s:
            korean_lines.append(s)
    return '\n'.join(korean_lines) if korean_lines else text.strip()


# ── 반박 프롬프트 ────────────────────────────────────────────────────────────

def _generate_search_query(topic: str, target_argument: str) -> str:
    """상대 논거에서 핵심 키워드를 추출해 반박 검색 쿼리를 생성한다."""
    # 볼드/마크다운 제거
    clean = re.sub(r'\*{1,2}', '', target_argument)
    # 핵심 주장 추출
    key = _extract_key_claim(clean)
    # 한국어 명사구만 추출 (조사/어미 제거는 하지 않고 길이로 자름)
    key = re.sub(r'[^\w가-힣\s]', '', key).strip()
    # 너무 길면 앞부분만
    words = key.split()
    if len(words) > 5:
        words = words[:5]
    query = ' '.join(words) + ' 반박 근거'
    return query[:40]


def _pre_search_rebuttal(topic: str, target_argument: str) -> Tuple[str, List[Dict]]:
    """연쇄논박용 사전검색. LLM이 생성한 쿼리로 팩트체크 검색."""
    tool_calls_log: List[Dict] = []

    query = _generate_search_query(topic, target_argument)
    tool_calls_log.append({"name": "search_web", "args": {"query": query}})
    web_result = search_web.invoke({"query": query})
    result = _truncate_tool_result(web_result)

    return result, tool_calls_log


def _extract_key_claim(speech: str) -> str:
    """상대 발언에서 핵심 주장 1문장을 추출한다."""
    # 결론 섹션 우선
    m = re.search(r'(?:결론|따라서|그러므로)[^\n]*', speech)
    if m:
        return m.group().strip()
    # 마지막 한국어 문장
    sentences = [s.strip() for s in speech.replace('\n', ' ').split('.') if s.strip() and re.search(r'[가-힣]', s)]
    if sentences:
        return sentences[-1] + '.'
    return speech[:100]


def _build_rebuttal_prompt(
    target_speech: str,
    target_display: str,
    stance_kr: str,
    search_results: str = "",
    my_previous: str = "",
    attack_style: str = "",
) -> str:
    context = ""
    if search_results:
        context += f"\n[참고 자료 — 반박 근거로 활용하라]\n{search_results}\n"
    if my_previous:
        context += f"\n[이전 발언 — 같은 내용 반복 금지]\n{my_previous}\n"

    return f"""상대 발언:
{target_speech}
{context}
상대 주장에서 틀린 부분을 찾아 반박하라. ({attack_style})

[규칙]
- 3~4문장으로만 답변
- 핵심에 **강조** 사용
- 소제목·번호·목록·볼드 번호(**1.** 등) 금지. 문장으로만 서술
- 반드시 합니다체(격식체). "~한다", "~이다" 금지. "~합니다", "~입니다"만 사용
- 한국어로 작성. 고유명사(기관명, 인명, 기술명)만 영어 허용
- 자체적으로 수치를 지어내지 마라. 검색 결과에 있는 수치만 인용 가능

반드시 아래 형식으로만 출력:

### 반박 시작
(반박 내용)
### 반박 끝"""


# ── 반박 생성 ────────────────────────────────────────────────────────────────

def _generate_rebuttal_speech(
    agent: Dict,
    prompt: str,
    target_display: str,
    stance: str,
) -> Tuple[str, str, List[Dict]]:
    """소형 모델 판단 + DeepSeek 생성. 필요할 때만 검색."""
    stance_kr = "찬성" if stance == "PRO" else "반대"
    system = (
        f"{agent['system_prompt']}\n\n"
        f"[최우선 규칙] 너는 {stance_kr} 입장이다. "
        f"반드시 3~4문장으로만 답변하라. "
        f"상대 주장의 오류만 공격하라. 자기 의견 피력은 마지막 1문장으로만."
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ]

    tool_calls_log: List[Dict] = []

    # DeepSeek으로 반박 생성
    response: AIMessage = _invoke_with_retry(_rebuttal_llm, messages, label="rebuttal")
    raw = response.content if isinstance(response.content, str) else str(response.content)
    speech = _postprocess_speech(_extract_rebuttal_text(raw))

    # CoT 유출 또는 무효 → 1회 재시도
    if not _is_valid_rebuttal(speech):
        logger.warning("[rebuttal] speech 무효, 재시도")
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content="한국어로만 3~4문장으로 반박하세요."))
        retry: AIMessage = _invoke_with_retry(_rebuttal_llm, messages, label="rebuttal_retry")
        raw = retry.content if isinstance(retry.content, str) else str(retry.content)
        speech = _postprocess_speech(_extract_rebuttal_text(raw))

    # 영어 잔재 감지 → LLM 수정 요청
    eng_words = re.findall(r'(?<![a-zA-Z])[a-z]{4,}(?![a-zA-Z])', speech)
    if eng_words:
        logger.warning("[rebuttal] 영어 감지: %s → 수정 요청", eng_words[:3])
        messages.append(AIMessage(content=raw))
        messages.append(HumanMessage(content=f'다음 영어 단어를 한국어로 바꿔서 다시 작성하라: {", ".join(eng_words[:5])}\n한국어만 사용. 같은 형식 유지.'))
        fix: AIMessage = _invoke_with_retry(_rebuttal_llm, messages, label="rebuttal_fix_eng")
        raw_fix = fix.content if isinstance(fix.content, str) else str(fix.content)
        fixed = _postprocess_speech(_extract_rebuttal_text(raw_fix))
        if _is_valid_rebuttal(fixed):
            speech = fixed
            raw = raw_fix

    # fallback
    if not _is_valid_rebuttal(speech):
        logger.warning("[rebuttal] fallback 사용")
        speech = (
            f"{target_display}의 주장은 핵심 전제가 부족합니다. "
            f"따라서 설득력이 없습니다. "
            f"저는 {stance_kr} 입장을 유지합니다."
        )

    return speech, raw, tool_calls_log


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


def _pick_one_argument(speech: str) -> str:
    """입론에서 논거 1 또는 논거 2를 랜덤으로 하나만 추출한다."""
    parts = re.split(r'###\s*논거\s*\d+\s*[:：]?', speech)
    arguments = []
    for i, p in enumerate(parts):
        if i == 0:
            continue  # 자기소개 부분 스킵
        # 결론 이후 제거
        conclusion_idx = p.find('### 결론')
        if conclusion_idx != -1:
            p = p[:conclusion_idx]
        text = p.strip()
        if text and len(text) > 20:
            arguments.append(text)
    if arguments:
        return random.choice(arguments)
    # 파싱 실패 시 원문 그대로 반환
    return speech


_ATTACK_STYLES = [
    "전제 공격: 상대 주장에 깔린 가정이 틀렸음을 지적하라",
    "현실성 공격: 실제 상황에서 작동하지 않는다는 점을 지적하라",
    "부작용 공격: 해당 주장으로 인해 발생하는 문제를 강조하라",
    "비교 공격: 더 나은 대안이 있음을 제시하라",
    "데이터 공격: 상대 근거의 신뢰성이나 부족함을 지적하라",
]

# 에이전트별 공격 방식 카운터 (같은 방식 반복 방지)
_attack_counter: Dict[str, int] = {}


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
    # 연쇄논박은 상대의 입론만 공격 (상대의 연쇄논박 발언이 아님)
    for entry in reversed(history):
        if entry["speaker_id"] == target_id and entry["phase"] == "opening":
            target_speech = entry["content"]
            target_stance = entry["stance"]
            break

    # 자신의 이전 발언 추출 (반복 방지)
    my_previous = ""
    for entry in reversed(history):
        if entry["speaker_id"] == agent["agent_id"] and entry["phase"] == "chained_rebuttal":
            my_previous = entry["content"][:200]
            break

    # 공격 방식 순환 할당
    aid = agent["agent_id"]
    idx = _attack_counter.get(aid, 0)
    attack_style = _ATTACK_STYLES[idx % len(_ATTACK_STYLES)]
    _attack_counter[aid] = idx + 1

    t_label = "찬성" if target_stance == "PRO" else "반대"
    target_display = f"{t_label}{target_stance_num}" if target_id != "user" else "사용자"
    stance_kr = "찬성" if agent["stance"] == "PRO" else "반대"

    # 상대 입론에서 논거 하나만 랜덤 추출
    target_argument = _pick_one_argument(target_speech)

    # 소형 모델이 검색 필요 여부 판단
    search_query = _decide_search(target_argument, attack_style)
    search_results = ""
    tool_calls_log: List[Dict] = []
    if search_query:
        logger.info("[rebuttal] 검색 판단: '%s'", search_query)
        tool_calls_log.append({"name": "search_web", "args": {"query": search_query}})
        web_result = search_web.invoke({"query": search_query})
        search_results = _truncate_tool_result(web_result)
    else:
        logger.info("[rebuttal] 검색 불필요 판단")

    # Qwen 7B로 약점 사전 분석
    weakness = analyze_weakness(target_argument, topic)
    if weakness:
        tool_calls_log.append({"name": "analyze_weakness", "result": weakness})
        logger.info("[rebuttal] 약점 분석: %s", weakness[:60])
        attack_style = f"{attack_style} — 특히 이 약점을 공격하라: {weakness}"

    prompt = _build_rebuttal_prompt(
        target_speech=target_argument,
        target_display=target_display,
        stance_kr=stance_kr,
        search_results=search_results,
        my_previous=my_previous,
        attack_style=attack_style,
    )

    speech, raw, _tool_log = _generate_rebuttal_speech(
        agent=agent, prompt=prompt,
        target_display=target_display, stance=agent["stance"],
    )

    return DebateEntry(
        turn=current_turn, speaker_id=agent["agent_id"],
        stance=agent["stance"], phase="chained_rebuttal",
        content=speech, target_id=target_id,
        tool_calls_log=tool_calls_log, json_raw=raw,
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

        # 공격만 수행 (응답 턴 제거)
        if attacker_id == "user":
            print(f"  [라운드 {round_num}] 공격: 사용자 → {target_id} (API 대기)\n")
            continue

        agent = agent_map[attacker_id]
        slabel = "찬성" if agent["stance"] == "PRO" else "반대"
        display = f"{slabel}{stance_nums[attacker_id]}"
        print(f"  [라운드 {round_num}] {display} → {target_id}")

        entry = generate_ai_rebuttal(
            topic=topic, history=history, agent=agent,
            target_id=target_id, stance_num=stance_nums[attacker_id],
            target_stance_num=stance_nums.get(target_id, 0),
            current_turn=current_turn, is_response=False,
        )
        history.append(entry)
        current_turn += 1
        pairs[pair_idx]["done"] = True
        print(f"  [라운드 {round_num}] 완료 (turn={entry['turn']})\n")

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
