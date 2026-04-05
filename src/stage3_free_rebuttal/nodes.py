"""
nodes.py — 3단계: 자유 논박(Free Rebuttal) 노드

[동작 흐름]
    1. speaking_order 순서대로 각 발언자의 차례
    2. 자기 차례에 skip 가능 (질문 받은 경우 제외)
    3. skip 안 하면 반대편 상대 지정 → 최대 4번 핑퐁
       sub-turn 1: 공격자 → 타겟 (공격/질문)
       sub-turn 2: 타겟 → 공격자 (응답, skip 불가)
       sub-turn 3: 공격자 → 타겟 (후속 공격)
       sub-turn 4: 타겟 → 공격자 (최종 응답, skip 불가)
    4. 사용자(user) 차례는 건너뛰고 API에서 처리

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
    _LLM_KWARGS,
)
from src.stage2_rebuttal.nodes import (
    _extract_rebuttal_text,
    build_agent_stance_nums,
)
from src.state import DebateEntry, DebateState

# ── 자유논박 전용 LLM ───────────────────────────────────────────────────────
_fr_llm = ChatOpenAI(**{**_LLM_KWARGS, "max_tokens": 512})

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


def _is_repetitive(new_text: str, prev_entries: List[DebateEntry], threshold: float = 0.7) -> bool:
    """이전 발언과 중복도 체크. threshold 이상이면 반복으로 판단."""
    new_words = set(re.findall(r'[가-힣]{2,}', new_text))
    for entry in prev_entries[-4:]:
        old_text = entry["content"]
        old_words = set(re.findall(r'[가-힣]{2,}', old_text))
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


# ── 타겟 자동 선정 ───────────────────────────────────────────────────────────

def _pick_target(
    history: List[DebateEntry],
    speaker_id: str,
    speaker_stance: str,
    agents: List[Dict],
) -> Optional[str]:
    """가장 최근에 발언한 상대 진영 발언자를 타겟으로 선택한다."""
    opposite = "CON" if speaker_stance == "PRO" else "PRO"
    for entry in reversed(history):
        if entry["stance"] == opposite and entry["speaker_id"] != speaker_id:
            return entry["speaker_id"]
    for a in agents:
        if a["stance"] == opposite:
            return a["agent_id"]
    return None


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


# ── 자유논박 프롬프트 ────────────────────────────────────────────────────────

def _build_free_rebuttal_prompt(
    target_speech: str,
    stance_kr: str,
    prev_exchange: str = "",
) -> str:
    """자유논박 프롬프트: 주장+반박 → 질문 형태."""
    exchange_block = ""
    if prev_exchange:
        exchange_block = f"\n[교환 기록]\n{prev_exchange}\n"

    return f"""상대: {target_speech[:200]}
{exchange_block}
너는 {stance_kr}이다. 반박 후 마지막에 질문 1개를 던져라.

좋은 예1) 탄소 배출은 여전히 증가하고 있습니다. 기술 혁신만으로 이를 해결할 수 있다는 근거는 부족합니다. 그렇다면 규제 없이 이 문제를 어떻게 해결하시겠습니까?
좋은 예2) 규제 없는 인프라 확충은 환경 비용을 사회에 전가합니다. 기업이 자발적으로 환경 보호에 나선 사례가 있습니까?

나쁜 예) 환경 규제가 필요합니다. (질문 없음)

이전 발언 반복 금지. 2~3문장."""


# ── 단일 발언 생성 ───────────────────────────────────────────────────────────

def _generate_free_rebuttal(
    agent: Dict,
    prompt: str,
    stance: str,
    prev_entries: Optional[List[DebateEntry]] = None,
) -> Tuple[str, str]:
    """단일 LLM 호출로 자유논박 발언 생성. 최대 3문장 (주장+반박+질문)."""
    stance_kr = "찬성" if stance == "PRO" else "반대"
    opposite_kr = "반대" if stance == "PRO" else "찬성"
    system = (
        f"{agent['system_prompt']}\n\n"
        f"[규칙] 너는 {stance_kr}이다. {opposite_kr} 주장 금지. "
        f"자유토론이다. 2~3문장만. 한국어만."
    )
    messages = [
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ]

    response: AIMessage = _invoke_with_retry(_fr_llm, messages, label="free_rebuttal")
    raw = response.content if isinstance(response.content, str) else str(response.content)
    speech = _postprocess_speech(_extract_rebuttal_text(raw))
    speech = _remove_self_repeat(speech)
    speech = _truncate_to_sentences(speech, max_sentences=3)

    def _get_fallback() -> str:
        msgs = _FALLBACK_MSGS.get(stance, _FALLBACK_MSGS["PRO"])
        idx = _fallback_idx.get(stance, 0)
        _fallback_idx[stance] = idx + 1
        return msgs[idx % len(msgs)]

    # 품질 체크
    korean_count = len(re.findall(r'[가-힣]', speech))
    total_count = len(speech.strip())
    is_junk = (
        not speech
        or total_count < 10
        or (total_count > 0 and korean_count / total_count < 0.3)
    )
    if is_junk:
        logger.warning("[free_rebuttal] fallback 사용")
        speech = _get_fallback()

    # 반복 체크
    if prev_entries and _is_repetitive(speech, prev_entries):
        logger.warning("[free_rebuttal] 반복 감지 → fallback")
        speech = _get_fallback()

    return speech, raw


# ── 핑퐁 교환 실행 ──────────────────────────────────────────────────────────

def run_pingpong_exchange(
    history: List[DebateEntry],
    attacker: Dict,
    defender: Dict,
    attacker_display: str,
    defender_display: str,
    current_turn: int,
    max_sub_turns: int = MAX_PINGPONG,
) -> Tuple[List[DebateEntry], int]:
    """공격자와 방어자 간 최대 max_sub_turns회 핑퐁 교환."""
    entries: List[DebateEntry] = []
    participants = [attacker, defender]
    prev_exchange = ""

    for sub_turn_idx in range(max_sub_turns):
        sub_turn_num = sub_turn_idx + 1
        is_attacker_turn = sub_turn_idx % 2 == 0
        current_agent = participants[sub_turn_idx % 2]
        target_agent = participants[(sub_turn_idx + 1) % 2]

        current_display = attacker_display if is_attacker_turn else defender_display
        target_display = defender_display if is_attacker_turn else attacker_display

        # 직전 발언 가져오기
        if entries:
            target_speech = entries[-1]["content"]
        else:
            target_speech = "(발언 기록 없음)"
            for entry in reversed(history):
                if entry["speaker_id"] == target_agent["agent_id"]:
                    target_speech = entry["content"]
                    break

        turn_label = "공격" if is_attacker_turn else "응답"
        print(f"    [{current_display}] sub-turn {sub_turn_num}/{max_sub_turns} ({turn_label}) → {target_display}")

        prompt = _build_free_rebuttal_prompt(
            target_speech=target_speech,
            stance_kr="찬성" if current_agent["stance"] == "PRO" else "반대",
            prev_exchange=prev_exchange,
        )

        speech, raw = _generate_free_rebuttal(
            agent=current_agent,
            prompt=prompt,
            stance=current_agent["stance"],
            prev_entries=entries,
        )

        entry = DebateEntry(
            turn=current_turn,
            speaker_id=current_agent["agent_id"],
            stance=current_agent["stance"],
            phase="free_rebuttal",
            content=speech,
            target_id=target_agent["agent_id"],
            tool_calls_log=[],
            json_raw=raw,
        )
        entries.append(entry)
        history.append(entry)
        current_turn += 1
        prev_exchange += f"\n[{current_display}] {speech}\n"

    return entries, current_turn


# ── 메인 노드 ─────────────────────────────────────────────────────────────────

def free_rebuttal_node(state: DebateState) -> DebateState:
    """3단계 자유 논박 노드."""
    _opening_mod._used_doc_ids = set()

    history: List[DebateEntry] = list(state["debate_history"])
    current_turn: int = state["current_turn"]
    agent_map = {a["agent_id"]: a for a in state["agents"]}
    speaking_order = state["speaking_order"]
    stance_nums = build_agent_stance_nums(state["agents"], speaking_order)

    print(f"\n[3단계: 자유 논박]\n")

    user_encountered = False
    for speaker_id in speaking_order:
        if speaker_id == "user":
            user_encountered = True
            print(f"  [사용자] 사용자 차례 → API 대기\n")
            continue

        agent = agent_map[speaker_id]
        attacker_display = _get_display_name(speaker_id, agent_map, stance_nums)

        target_id = _pick_target(history, speaker_id, agent["stance"], state["agents"])
        if target_id is None:
            logger.warning("[free_rebuttal] %s의 반박 대상을 찾을 수 없음", speaker_id)
            continue

        if target_id == "user":
            target_display = "사용자"
            print(f"  [{attacker_display}] → {target_display} (단일 공격)")

            target_speech = "(발언 기록 없음)"
            for entry in reversed(history):
                if entry["speaker_id"] == "user":
                    target_speech = entry["content"]
                    break

            prompt = _build_free_rebuttal_prompt(
                target_speech=target_speech,
                stance_kr="찬성" if agent["stance"] == "PRO" else "반대",
            )
            speech, raw = _generate_free_rebuttal(agent, prompt, agent["stance"], prev_entries=[])

            history.append(DebateEntry(
                turn=current_turn, speaker_id=speaker_id,
                stance=agent["stance"], phase="free_rebuttal",
                content=speech, target_id="user",
                tool_calls_log=[], json_raw=raw,
            ))
            current_turn += 1
            print(f"    완료 (turn={current_turn - 1})\n")
            continue

        # AI vs AI: 핑퐁
        defender = agent_map[target_id]
        defender_display = _get_display_name(target_id, agent_map, stance_nums)
        print(f"  [{attacker_display}] ↔ {defender_display} 핑퐁 교환")

        _, current_turn = run_pingpong_exchange(
            history=history,
            attacker=agent,
            defender=defender,
            attacker_display=attacker_display,
            defender_display=defender_display,
            current_turn=current_turn,
        )
        print(f"  [{attacker_display}] ↔ {defender_display} 완료\n")

    if user_encountered:
        next_phase = "free_rebuttal"
        print(f"[3단계: 자유 논박] AI 발언 완료 → 사용자 발언 대기\n")
    else:
        next_phase = "role_reversal"
        print(f"[3단계: 자유 논박] 완료 → 4단계로 전환\n")

    return DebateState(**{
        **state,
        "debate_history": history,
        "current_turn": current_turn,
        "phase": next_phase,
    })
