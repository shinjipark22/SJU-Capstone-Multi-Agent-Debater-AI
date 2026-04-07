"""
nodes.py — 3단계: 자유 논박(Free Rebuttal) 노드

[동작 흐름]
    1. 사용자 vs 선택된 상대 에이전트 1:1 핑퐁
    2. 매 호출마다 에이전트가 1회 발언 생성
    3. 사용자 발언은 API에서 history에 삽입 후 재호출
    4. selected_opponent_id가 없으면 ValueError 발생

[설계 노트]
    - delimiter-free, 한국어 추출 방식
    - DeepSeek-R1-Distill-Qwen-14B 최적화
    - 도구 호출 없이 단일 LLM 호출
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

import src.stage1_opening.nodes as _opening_mod
from src.stage1_opening.nodes import (
    _invoke_with_retry,
    _postprocess_speech,
    _truncate_tool_result,
    _remove_english_blocks,
    _has_cot_leakage,
    search_web,
    _LLM_KWARGS,
)
from src.stage2_rebuttal.nodes import (
    _extract_rebuttal_text,
    _decide_search,
    build_agent_stance_nums,
)
from src.state import DebateEntry, DebateState

# ── 자유논박 전용 LLM ───────────────────────────────────────────────────────
_fr_llm = ChatOpenAI(**{**_LLM_KWARGS, "max_tokens": 512, "temperature": 0.75})

MAX_PINGPONG = 4


# ── 자유논박 전용 후처리 ────────────────────────────────────────────────────────

def _truncate_to_sentences(text: str, max_sentences: int = 2) -> str:
    """한국어 문장 기준 max_sentences개까지만 남긴다."""
    # CJK 전각 문장부호 → 반각 변환
    text = text.replace('，', ',').replace('。', '.').replace('；', ';')
    text = text.replace('：', ':')
    # CJK 기호·전각문자 제거
    text = re.sub(r'[\u3000-\u303f\uff00-\uffef]', '', text)
    # 앞쪽 쓰레기 문자 제거 (한글 시작 전의 구두점/따옴표/공백)
    text = re.sub(r'^[^가-힣]*(?=[가-힣])', '', text.strip())
    # "상대:" 프롬프트 유출 제거
    text = re.sub(r'^상대\s*[:：]\s*', '', text.strip())
    # 메타 설명 유출 제거
    text = re.sub(r'^(찬성|반대)\s*토론자입니다[.]?\s*', '', text)
    text = re.sub(r'사용자가\s*제공한\s*(문맥|콘텍스트)에\s*따르면[,.]?\s*', '', text)
    text = re.sub(r'(찬성|반대)\s*토론자로서\s*', '', text)
    text = re.sub(r'사용자는\s*(찬성|반대)자로서[,.]?\s*[^.]*[.]?\s*', '', text)
    # [반박] [질문] 등 태그 제거 (괄호 유무 모두)
    text = re.sub(r'\[?(반박|질문|형식|찬성자?\s*반박|반대자?\s*반박)\]?\s*\n?', '', text)
    text = re.sub(r'\[?(찬성|반대)\s*에이전트\d*\]?\s*\n?', '', text)
    # 형식 지시문 유출 제거
    text = re.sub(r'^반박\s*\d*~?\d*문장.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+~?\d*문장\s*\+?\s*\d*개?\s*', '', text)
    text = re.sub(r'예\s*[:：]\s*"[^"]*"\s*', '', text)
    # 프롬프트 지시문 누출 제거 (~입니다/~습니다 체 등)
    text = re.sub(r'~?입니다/~?습니다\s*체?[.]?\s*', '', text)
    text = re.sub(r'핵심에\s*\*{0,2}강조\*{0,2}[.]?\s*', '', text)
    text = re.sub(r'일반적으로\s*~로\s*알려져\s*있다[.]?\s*', '', text)
    # 영어 전용 줄 제거 (한글 없는 줄만)
    lines = text.split('\n')
    text = '\n'.join(l for l in lines if not l.strip() or re.search(r'[가-힣]', l))
    # 연속 구두점/쓰레기 제거
    text = re.sub(r'[,.\s]{3,}', ' ', text)
    text = re.sub(r'[""\'"]{2,}', '', text)
    # 잘린 문장 제거 (한국어 종결어미 없이 끝나는 마지막 토큰)
    text = re.sub(r'[가-힣]{1,5}[.]$', lambda m: m.group() if len(m.group()) > 3 else '', text)

    # 문장 분리 (.!? 뒤 공백 또는 줄바꿈)
    sentences = re.split(r'(?<=[.!?다])\s+', text.strip())
    kept = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # 한글이 포함된 문장만
        if not re.search(r'[가-힣]', s):
            continue
        # 너무 짧은 문장 (10자 미만) 제거
        if len(s) < 10:
            continue
        # 잘린 문장 감지: 조사/접속사로 끝나면 제거
        if re.search(r'[을를이가은는에와과의로며지만고서][\.,]?$', s):
            continue
        # 메타 설명 문장 제거
        if re.search(r'(주장하고\s*있습니다|강화시켜야\s*합니다|문맥에\s*따르면|답변을\s*작성|설명해야\s*합니다|논제에\s*대해|답변하기\s*위해)', s):
            continue
        kept.append(s)
        if len(kept) >= max_sentences:
            break
    # 마지막 문장이 .!?로 끝나지 않으면 잘린 것 → 제거
    while kept and not re.search(r'[.!?]$', kept[-1].strip()):
        kept.pop()
    result = ' '.join(kept)
    return result


def _is_repetitive(
    new_text: str,
    prev_entries: List[DebateEntry],
    threshold: float = 0.7,
    topic: str = "",
) -> bool:
    """이전 발언과 중복도 체크. 토픽 키워드는 겹침 계산에서 제외."""
    topic_words = set(re.findall(r'[가-힣]{2,}', topic)) if topic else set()
    new_words = set(re.findall(r'[가-힣]{2,}', new_text)) - topic_words
    for entry in prev_entries[-4:]:
        old_words = set(re.findall(r'[가-힣]{2,}', entry["content"])) - topic_words
        if not new_words or not old_words:
            continue
        overlap = len(new_words & old_words) / max(len(new_words | old_words), 1)
        if overlap > threshold:
            return True
    return False


def _remove_self_repeat(text: str) -> str:
    """같은/유사 문장이 반복되면 한 번만 남긴다."""
    sentences = re.split(r'(?<=[.!?다])\s+', text.strip())
    seen = []
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        # 기존 문장과 단어 겹침 체크
        s_words = set(re.findall(r'[가-힣]{2,}', s))
        is_dup = False
        for prev in seen:
            prev_words = set(re.findall(r'[가-힣]{2,}', prev))
            if s_words and prev_words:
                overlap = len(s_words & prev_words) / max(len(s_words | prev_words), 1)
                if overlap > 0.6:
                    is_dup = True
                    break
        if not is_dup:
            seen.append(s)
    return ' '.join(seen)


# 다양한 fallback 메시지 (주장 + 질문 형태)
# 범용 fallback 메시지 (주제 무관)
_FALLBACK_MSGS = {
    "PRO": [
        "상대의 주장은 현실적 근거가 부족합니다. 그렇다면 이를 뒷받침할 구체적 사례를 제시할 수 있습니까?",
        "상대의 논리는 핵심 전제가 검증되지 않았습니다. 해당 전제가 성립한다는 근거가 있습니까?",
        "상대의 주장이 실현 가능하다는 증거가 부족합니다. 실제로 성공한 사례가 있습니까?",
    ],
    "CON": [
        "상대의 주장은 이론적으로는 타당하지만 현실 적용에 한계가 있습니다. 실행 가능한 대안이 있습니까?",
        "상대의 논거는 부작용을 간과하고 있습니다. 이로 인한 부정적 영향을 어떻게 해결하시겠습니까?",
        "상대의 접근은 단기적 효과에 머무릅니다. 장기적으로도 유효하다는 근거가 있습니까?",
    ],
}
_fallback_idx: Dict[str, int] = {}


# ── 표시명 유틸리티 ──────────────────────────────────────────────────────────

def _get_display_name(
    speaker_id: str,
    agent_map: Dict[str, Dict],
    stance_nums: Dict[str, int],
) -> str:
    if speaker_id == "user":
        return "사용자"
    if speaker_id in agent_map:
        label = "찬성" if agent_map[speaker_id]["stance"] == "PRO" else "반대"
        return f"{label} 에이전트{stance_nums.get(speaker_id, 0)}"
    return speaker_id


# ── 검색 (Qwen2.5-1.5B 조건부 검색 — 연쇄논박과 동일) ──────────────────────

def _search_for_response(topic: str, target_speech: str) -> Tuple[str, List[Dict]]:
    """Qwen2.5-1.5B가 검색 필요 여부를 판단하고, 필요 시 검색한다."""
    tool_calls_log: List[Dict] = []
    query = _decide_search(target_speech, "")
    if not query:
        return "", tool_calls_log
    tool_calls_log.append({"name": "search_web", "args": {"query": query}})
    web_result = search_web.invoke({"query": query})
    return _truncate_tool_result(web_result, 200), tool_calls_log


# ── 자유논박 프롬프트 ────────────────────────────────────────────────────────

def _build_free_rebuttal_prompt(
    target_speech: str,
    stance_kr: str,
    topic: str = "",
    prev_exchange: str = "",
    used_arguments: str = "",
    used_questions: str = "",
    search_result: str = "",
) -> str:
    """자유논박 프롬프트: 답변 → 반박 → 질문."""
    exchange_block = ""
    if prev_exchange:
        exchange_block = f"\n[이전 교환]\n{prev_exchange}\n"

    used_block = ""
    if used_arguments:
        used_block = f"\n[이미 사용한 논점 — 반복 금지]\n{used_arguments}\n"

    ref_block = ""
    if search_result:
        ref_block = f"\n[참고 자료 — 답변 근거로만 활용]\n{search_result}\n"

    # 입장에 따른 핵심 주장 설명
    opposite_kr = "반대" if stance_kr == "찬성" else "찬성"
    if stance_kr == "찬성":
        my_position = f"'{topic}'에 찬성한다"
    else:
        my_position = f"'{topic}'에 반대한다"

    # Step 1 프롬프트: 답변 + 반박
    step1 = f"""[토론 주제] {topic}
[내 입장] {my_position}

상대: {target_speech[:250]}
{exchange_block}{used_block}{ref_block}
[규칙]
- 반드시 내 입장({stance_kr})을 유지하라. 상대 입장에 동조 금지.
- 첫 문장에서 상대 질문에 직접 답하라
- 둘째 문장에서 상대 논리의 약점을 공격하라
- 참고 자료가 있으면 수치/사례를 1개 인용하라
- 일반론 금지, 구체적으로

1~2문장만. 한국어로 작성. 고유명사(기관명, 인명, 기술명)만 영어 허용.

반드시 아래 형식으로만 출력:

### 반박 시작
(반박 내용)
### 반박 끝
"""

    # Step 2 프롬프트: 질문 생성
    used_q_block = ""
    if used_questions:
        used_q_block = f"\n[이미 던진 질문 — 같은 질문 금지]\n{used_questions}\n"

    step2_template = """[내 입장] {my_position}

[내 발언]
{my_response}

[상대 발언]
{target_speech}
{used_q_block}
상대({opposite_kr})의 논리적 허점을 찌르는 질문을 1개만 만들어라.
내 입장({stance_kr})을 강화하는 방향이어야 한다. 이전에 던진 질문과 다른 새로운 질문이어야 한다.
반드시 ?로 끝나는 한 문장. 한국어만.""".replace(
        "{opposite_kr}", opposite_kr
    ).replace(
        "{my_position}", my_position
    ).replace(
        "{used_q_block}", used_q_block
    )

    return step1, step2_template


# ── 2-Step 발언 생성 ─────────────────────────────────────────────────────────

def _generate_free_rebuttal(
    agent: Dict,
    step1_prompt: str,
    step2_template: str,
    target_speech: str,
    stance: str,
    prev_entries: Optional[List[DebateEntry]] = None,
    topic: str = "",
) -> Tuple[str, str]:
    """2-Step 자유논박 발언 생성.

    Step 1: 답변+반박 (1~2문장)
    Step 2: 압박 질문 (1문장)
    결합하여 최종 발언.
    """
    stance_kr = "찬성" if stance == "PRO" else "반대"
    opposite_kr = "반대" if stance == "PRO" else "찬성"
    system = (
        f"{agent['system_prompt']}\n\n"
        f"너는 {stance_kr} 토론자다. {opposite_kr} 입장 절대 금지.\n"
        f"실제 토론처럼 날카롭고 구체적으로 말하라."
    )

    def _get_fallback() -> str:
        msgs = _FALLBACK_MSGS.get(stance, _FALLBACK_MSGS["PRO"])
        idx = _fallback_idx.get(stance, 0)
        _fallback_idx[stance] = idx + 1
        return msgs[idx % len(msgs)]

    def _clean(text: str) -> str:
        text = _postprocess_speech(_extract_rebuttal_text(text))
        text = _remove_self_repeat(text)
        text = _truncate_to_sentences(text, max_sentences=2)
        return text

    # ── Step 1: 답변 + 반박
    messages1 = [
        SystemMessage(content=system),
        HumanMessage(content=step1_prompt),
    ]
    response1: AIMessage = _invoke_with_retry(_fr_llm, messages1, label="free_rebuttal_step1")
    raw1 = response1.content if isinstance(response1.content, str) else str(response1.content)
    rebuttal = _clean(raw1)

    # Step 1 품질 체크 (CoT 유출 감지 포함)
    korean_count = len(re.findall(r'[가-힣]', rebuttal))
    total_count = len(rebuttal.strip())
    if not rebuttal or total_count < 10 or (total_count > 0 and korean_count / total_count < 0.3):
        logger.warning("[free_rebuttal] step1 품질 불량 → fallback")
        return _get_fallback(), raw1
    if _has_cot_leakage(rebuttal):
        logger.warning("[free_rebuttal] step1 CoT 유출 → 재생성")
        response1_cot: AIMessage = _invoke_with_retry(_fr_llm, messages1, label="free_rebuttal_step1_cot_retry")
        raw1 = response1_cot.content if isinstance(response1_cot.content, str) else str(response1_cot.content)
        rebuttal = _clean(raw1)

    # 반복 체크
    if prev_entries and _is_repetitive(rebuttal, prev_entries, topic=topic):
        logger.warning("[free_rebuttal] step1 반복 감지 → fallback")
        return _get_fallback(), raw1

    # 입장 혼동 체크: 상대 입장에 완전히 동조하는 경우만 (부분 인정은 허용)
    if re.search(r'상대의?\s*(주장|의견)에\s*(전적으로\s*)?(동의|공감)합니다', rebuttal):
        logger.warning("[free_rebuttal] step1 입장 혼동 감지 → 재생성")
        response1_retry: AIMessage = _invoke_with_retry(_fr_llm, messages1, label="free_rebuttal_step1_retry")
        raw1_retry = response1_retry.content if isinstance(response1_retry.content, str) else str(response1_retry.content)
        rebuttal = _clean(raw1_retry)
        raw1 = raw1_retry

    # ── Step 2: 질문 생성
    step2_prompt = step2_template.format(
        stance_kr=stance_kr,
        my_response=rebuttal,
        target_speech=target_speech[:150],
    )
    messages2 = [
        SystemMessage(content=f"너는 {stance_kr} 토론자다. 질문만 생성하라."),
        HumanMessage(content=step2_prompt),
    ]
    response2: AIMessage = _invoke_with_retry(_fr_llm, messages2, label="free_rebuttal_step2")
    raw2 = response2.content if isinstance(response2.content, str) else str(response2.content)
    question = _postprocess_speech(_extract_rebuttal_text(raw2)).strip()

    # 질문이 ?로 끝나는 한국어 문장인지 확인
    q_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', question) if s.strip().endswith('?') and re.search(r'[가-힣]', s)]
    if q_sentences:
        question = q_sentences[-1]  # 마지막 질문만
    else:
        question = "이에 대한 구체적 근거를 제시할 수 있습니까?"

    # ── 결합
    speech = f"{rebuttal} {question}"
    raw = f"[STEP1]\n{raw1}\n\n[STEP2]\n{raw2}"

    return speech, raw


# ── 메인 노드 (1:1 핑퐁) ──────────────────────────────────────────────────────

def free_rebuttal_node(state: DebateState) -> DebateState:
    """3단계 자유 논박 노드.

    사용자 vs 선택된 상대 에이전트 1:1 핑퐁.
    매 호출마다 에이전트가 1회 발언을 생성하고, 사용자 입력을 기다린다.
    """
    _opening_mod._used_doc_ids = set()

    # ── 상대 에이전트 확인
    selected_id = state.get("selected_opponent_id")
    if not selected_id:
        raise ValueError(
            "[free_rebuttal] selected_opponent_id가 필요합니다. "
            "자유논박 시작 전에 사용자가 상대 에이전트를 선택해야 합니다."
        )

    agent_map = {a["agent_id"]: a for a in state["agents"]}
    if selected_id not in agent_map:
        raise ValueError(f"[free_rebuttal] 존재하지 않는 에이전트: {selected_id}")

    opponent = agent_map[selected_id]
    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]
    speaking_order = state["speaking_order"]
    stance_nums = build_agent_stance_nums(state["agents"], speaking_order)

    opponent_display = _get_display_name(selected_id, agent_map, stance_nums)
    print(f"\n[3단계: 자유 논박] 사용자 ↔ {opponent_display}\n")

    # ── 사용자의 최근 발언 찾기
    user_speech = "(발언 기록 없음)"
    for entry in reversed(history):
        if entry["speaker_id"] == "user":
            user_speech = entry["content"]
            break

    # ── 이전 자유논박 교환 기록 구성
    prev_exchange = ""
    fr_entries = [e for e in history if e["phase"] == "free_rebuttal"]
    for e in fr_entries[-4:]:  # 최근 4턴만 (너무 길면 모델 혼란)
        speaker = "사용자" if e["speaker_id"] == "user" else opponent_display
        prev_exchange += f"[{speaker}] {e['content']}\n"

    # ── 이미 사용한 논점 추출 (에이전트 이전 발언에서 핵심 키워드)
    used_arguments = ""
    agent_prev = [e for e in history if e["speaker_id"] == selected_id and e["phase"] == "free_rebuttal"]
    if agent_prev:
        used_points = []
        for e in agent_prev:
            # 각 발언에서 첫 문장만 요약으로 사용
            first_sent = e["content"].split('.')[0] + '.'
            if len(first_sent) > 15:
                used_points.append(f"- {first_sent[:60]}")
        used_arguments = "\n".join(used_points[-3:])  # 최근 3개만

    # ── 이전에 던진 질문 추출 (Step 2 반복 방지용)
    used_questions = ""
    if agent_prev:
        prev_qs = []
        for e in agent_prev:
            sentences = re.split(r'(?<=[.!?])\s+', e["content"])
            for s in sentences:
                if s.strip().endswith('?'):
                    prev_qs.append(f"- {s.strip()}")
        if prev_qs:
            used_questions = "\n".join(prev_qs[-3:])

    # 디버그
    if used_arguments:
        print(f"  [디버그] 이미 사용한 논점:\n{used_arguments}\n")
    else:
        print(f"  [디버그] 이전 발언 없음 — 첫 턴\n")

    # ── 조건부 검색: Qwen2.5-1.5B가 검색 필요 여부 판단
    search_result = ""
    tool_calls_log: List[Dict] = []
    search_result, tool_calls_log = _search_for_response(
        topic=state["topic"], target_speech=user_speech,
    )
    if tool_calls_log:
        print(f"  [검색] Qwen2.5-1.5B 판단 → 웹 검색 실행\n")
    else:
        print(f"  [검색] Qwen2.5-1.5B 판단 → 검색 불필요\n")

    # ── 에이전트 발언 생성 (2-Step)
    stance_kr = "찬성" if opponent["stance"] == "PRO" else "반대"
    step1_prompt, step2_template = _build_free_rebuttal_prompt(
        target_speech=user_speech,
        stance_kr=stance_kr,
        topic=state["topic"],
        prev_exchange=prev_exchange,
        used_arguments=used_arguments,
        used_questions=used_questions,
        search_result=search_result,
    )

    prev_entries = [e for e in history if e["speaker_id"] == selected_id and e["phase"] == "free_rebuttal"]
    speech, raw = _generate_free_rebuttal(
        agent=opponent,
        step1_prompt=step1_prompt,
        step2_template=step2_template,
        target_speech=user_speech,
        stance=opponent["stance"],
        prev_entries=prev_entries,
        topic=state["topic"],
    )

    # ── 발언 기록
    history.append(DebateEntry(
        turn=current_turn,
        speaker_id=selected_id,
        stance=opponent["stance"],
        phase="free_rebuttal",
        content=speech,
        target_id="user",
        tool_calls_log=tool_calls_log,
        json_raw=raw,
    ))
    current_turn += 1

    print(f"  [{opponent_display}] → 사용자 (turn={current_turn - 1})")
    print(f"  {speech}\n")
    print(f"[3단계: 자유 논박] 에이전트 발언 완료 → 사용자 발언 대기\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "phase": "free_rebuttal",
    })
