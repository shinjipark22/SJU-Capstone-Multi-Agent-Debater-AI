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

import src.phase1.stage1_opening.nodes as _opening_mod
from src.phase1.stage1_opening.nodes import _remove_english_blocks


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

# ── graph/ 모듈에서 Searcher 함수 임포트 ──────────────────────────────────
from src.graph.searcher import search_web


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


# (기존 _generate_rebuttal_speech, _build_rebuttal_prompt 등은
#  search_write_review 서브그래프가 대체하므로 삭제됨)


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
    """연쇄논박 발언 생성 — search_write_review 서브그래프 사용."""
    from src.graph.subgraphs import search_write_review

    target_speech = "(발언 기록 없음)"
    target_stance = "CON" if agent["stance"] == "PRO" else "PRO"
    for entry in reversed(history):
        if entry["speaker_id"] == target_id and entry["phase"] == "opening":
            target_speech = entry["content"]
            target_stance = entry["stance"]
            break

    target_argument = _pick_one_argument(target_speech)

    result = search_write_review.invoke({
        "topic": topic,
        "agent": agent,
        "expected_stance": agent["stance"],
        "target_argument": target_argument,
        "my_opening": "",
        "opp_opening": "",
        "chain": [],
        "prev_weaknesses": "",
        "prev_attacks": "",
        "mode": "rebuttal",
        "weakness": "",
        "search_results": "",
        "search_query": "",
        "speech": "",
        "raw": "",
        "review_result": {},
        "retry_count": 0,
        "tool_calls_log": [],
    })

    speech = result["speech"]
    raw = result["raw"]
    tool_calls_log = result.get("tool_calls_log", [])

    # fallback
    if not speech or len(speech.strip()) < 15:
        t_label = "찬성" if target_stance == "PRO" else "반대"
        target_display = f"{t_label}{target_stance_num}" if target_id != "user" else "사용자"
        stance_kr = "찬성" if agent["stance"] == "PRO" else "반대"
        speech = f"{target_display}의 주장은 핵심 전제가 부족합니다. 저는 {stance_kr} 입장을 유지합니다."

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
