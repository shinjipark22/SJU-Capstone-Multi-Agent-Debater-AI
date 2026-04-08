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
    _check_stance,
    _generate_attack_question,
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
    # 재시도 프롬프트 유출 제거
    text = re.sub(r'한국어로만\s*으?로?\s*(?:하고|으로)?\s*하세요[.]?\s*', '', text)
    text = re.sub(r'한국어로만\s*\d*~?\d*문장으로\s*반박하고\s*질문하세요[.]?\s*', '', text)
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

    # 영어 CoT 잔재 제거 (문장 앞뒤 영어 구간)
    text = re.sub(r'^[a-zA-Z\s,.\'"():;!?]+(?=[가-힣])', '', text)  # 앞쪽 영어
    text = re.sub(r'(?<=[.!?])\s*[a-zA-Z\s,.\'"():;!?]{20,}$', '', text)  # 뒤쪽 영어 (20자 이상)
    text = re.sub(r'"[^"]*".*?(?:translates?|means?|refers?).*?[.]\s*', '', text)  # "..." translates to 패턴

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
        # 영어 비율이 50% 이상인 문장 제거
        kor = len(re.findall(r'[가-힣]', s))
        eng = len(re.findall(r'[a-zA-Z]', s))
        if kor + eng > 0 and eng / (kor + eng) > 0.5:
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

def _build_system_prompt(
    agent: Dict,
    stance: str,
    topic: str,
    my_opening: str = "",
    opp_opening: str = "",
    search_result: str = "",
) -> str:
    """시스템 프롬프트: 입론 + 참고 자료를 포함한 정적 컨텍스트."""
    stance_kr = "찬성" if stance == "PRO" else "반대"
    opposite_kr = "반대" if stance == "PRO" else "찬성"

    return (
        f"너는 세계 최고 수준의 {stance_kr} 토론자다. "
        f"토론 주제: {topic}\n"
        f"{opposite_kr} 입장 절대 금지. 반드시 {stance_kr} 입장을 유지하라.\n"
        f"상대 발언에 반박하라. 2~3문장. 합니다체. 한국어만."
    )


def _build_conversation_messages(
    system: str,
    history: List[DebateEntry],
    agent_id: str,
    max_turns: int = 8,
) -> List:
    """debate_history를 실제 대화 메시지 리스트로 변환한다."""
    messages = [SystemMessage(content=system)]

    fr_entries = [e for e in history if e["phase"] == "free_rebuttal"]
    window = fr_entries[-max_turns:]

    for entry in window:
        content = entry["content"]
        if entry["speaker_id"] == agent_id:
            messages.append(AIMessage(content=content))
        else:
            messages.append(HumanMessage(content=content))

    return messages


# ── 자유논박 발언 생성 (챗봇 패턴) ──────────────────────────────────────────

def _generate_free_rebuttal(
    agent: Dict,
    agent_id: str,
    stance: str,
    topic: str,
    history: List[DebateEntry],
    search_result: str = "",
    my_opening: str = "",
    opp_opening: str = "",
    prev_entries: Optional[List[DebateEntry]] = None,
) -> Tuple[str, str]:
    """챗봇 패턴 자유논박: 대화 히스토리를 HumanMessage/AIMessage로 변환."""

    def _get_fallback() -> str:
        msgs = _FALLBACK_MSGS.get(stance, _FALLBACK_MSGS["PRO"])
        idx = _fallback_idx.get(stance, 0)
        _fallback_idx[stance] = idx + 1
        return msgs[idx % len(msgs)]

    def _clean(text: str) -> str:
        # <think> 제거만
        text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)
        text = text.replace('</think>', '').strip()
        # 외국 문자 제거 (한자, 일본어, 러시아어 등)
        text = re.sub(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff\u3040-\u309f\u30a0-\u30ff\u0400-\u04ff\u0e00-\u0e7f\u0600-\u06ff]+', '', text)
        # 영어 전용 줄만 제거 (한글 없는 줄)
        lines = text.split('\n')
        text = '\n'.join(l for l in lines if not l.strip() or re.search(r'[가-힣]', l))
        return text.strip()

    # 시스템 프롬프트 구성
    system = _build_system_prompt(
        agent, stance, topic, my_opening, opp_opening, search_result,
    )

    # 대화 히스토리를 메시지 리스트로 변환
    messages = _build_conversation_messages(system, history, agent_id)

    # 마지막이 HumanMessage인지 확인 (사용자 발언이 마지막이어야 함)
    if not messages or not isinstance(messages[-1], HumanMessage):
        for entry in reversed(history):
            if entry["speaker_id"] == "user":
                messages.append(HumanMessage(content=entry["content"]))
                break

    # LLM 호출 + 영어/CoT 유출 시 재시도 (최대 2회)
    response: AIMessage = _invoke_with_retry(_fr_llm, messages, label="free_rebuttal")
    raw = response.content if isinstance(response.content, str) else str(response.content)
    speech = _clean(raw)

    for retry_idx in range(3):
        korean_count = len(re.findall(r'[가-힣]', speech))
        total_count = len(speech.strip())
        is_bad = not speech or total_count < 10 or (total_count > 0 and korean_count / total_count < 0.3)
        is_cot = _has_cot_leakage(speech) if speech else False

        if not is_bad and not is_cot:
            break

        reason = "CoT 유출" if is_cot else "영어/빈 응답"
        logger.warning("[free_rebuttal] %s → 재시도 %d/3", reason, retry_idx + 1)
        # 영어 응답을 대화에 쌓지 않고 마지막 HumanMessage만 유지하며 한국어 강제
        last_human = None
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last_human = m.content
                break
        messages = [messages[0]]  # SystemMessage만 유지
        if last_human:
            messages.append(HumanMessage(content=f"{last_human}\n\n반드시 한국어로만 답하라. 2~3문장."))
        retry: AIMessage = _invoke_with_retry(_fr_llm, messages, label=f"free_rebuttal_retry{retry_idx}")
        raw = retry.content if isinstance(retry.content, str) else str(retry.content)
        speech = _clean(raw)

    # 최종 품질 체크
    korean_count = len(re.findall(r'[가-힣]', speech))
    total_count = len(speech.strip())
    if not speech or total_count < 10 or (total_count > 0 and korean_count / total_count < 0.3):
        logger.warning("[free_rebuttal] 최종 품질 불량 → fallback")
        return _get_fallback(), raw

    # 입장 판별: Qwen2.5-1.5B가 발언이 자기 진영인지 확인
    stance_kr = "찬성" if stance == "PRO" else "반대"
    if not _check_stance(speech, stance, topic):
        logger.warning("[free_rebuttal] 입장 혼동 감지 → 재생성")
        messages_retry = list(messages)  # 복사
        messages_retry.append(AIMessage(content=raw))
        messages_retry.append(HumanMessage(content=f"너는 {stance_kr} 입장이다. 상대 입장에 동의하지 마라. 다시 반박하라."))
        retry_stance: AIMessage = _invoke_with_retry(_fr_llm, messages_retry, label="free_rebuttal_stance_retry")
        raw = retry_stance.content if isinstance(retry_stance.content, str) else str(retry_stance.content)
        speech = _clean(raw)

    # 반복 체크
    if prev_entries and _is_repetitive(speech, prev_entries, topic=topic):
        logger.warning("[free_rebuttal] 반복 감지 → fallback")
        return _get_fallback(), raw

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

    # ── 입론 추출
    my_opening = ""
    opp_opening = ""
    for e in history:
        if e["phase"] == "opening" and e["speaker_id"] == selected_id:
            my_opening = e["content"][:300]
        if e["phase"] == "opening" and e["speaker_id"] == "user":
            opp_opening = e["content"][:300]

    # ── 조건부 검색: Qwen2.5-1.5B
    search_result = ""
    tool_calls_log: List[Dict] = []
    search_result, tool_calls_log = _search_for_response(
        topic=state["topic"], target_speech=user_speech,
    )
    if tool_calls_log:
        print(f"  [검색] Qwen2.5-1.5B → 검색 실행\n")
    else:
        print(f"  [검색] Qwen2.5-1.5B → 검색 불필요\n")

    # ── 에이전트 발언 생성 (챗봇 패턴)
    prev_entries = [e for e in history if e["speaker_id"] == selected_id and e["phase"] == "free_rebuttal"]
    speech, raw = _generate_free_rebuttal(
        agent=opponent,
        agent_id=selected_id,
        stance=opponent["stance"],
        topic=state["topic"],
        history=history,
        search_result=search_result,
        my_opening=my_opening,
        opp_opening=opp_opening,
        prev_entries=prev_entries,
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
