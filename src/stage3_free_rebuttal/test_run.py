"""
test_run.py — 3단계 자유논박(Free Rebuttal) 노드 독립 실행 테스트

터미널에서 실행:
    python -m src.stage3_free_rebuttal.test_run
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.phase0.persona_factory import create_agents
from src.state import AgentSnapshot, build_initial_state
from src.stage1_opening.nodes import opening_arguments_node
from src.stage2_rebuttal.nodes import chained_rebuttal_node
from src.stage3_free_rebuttal.nodes import free_rebuttal_node


# ── 테스트 설정 ───────────────────────────────────────────────────────────────

TOPIC_ID = "tech_001"
DEBATE_FORMAT = "2:2"
USER_STANCE = "PRO"
USER_INTENSITY = 3
AGENT_INTENSITIES = [3, 2, 4]
MAX_CYCLE = 2  # 테스트용 사이클 수

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "topics_20260323_processed.json"


def _load_topic(topic_id: str) -> dict:
    with _DATA_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    for category_topics in data.get("categories", {}).values():
        for t in category_topics:
            if t["id"] == topic_id:
                return t
    raise ValueError(f"topic ID '{topic_id}'를 찾을 수 없습니다.")


def _print_entries(entries, phase_label):
    """발언 목록을 포맷팅하여 출력한다."""
    for entry in entries:
        stance = entry["stance"]
        s_label = "찬성" if stance == "PRO" else "반대"

        print("-" * 70)
        print(f"  발언자  : {entry['speaker_id']} ({s_label})")
        print(f"  턴 번호 : {entry['turn']}")
        print(f"  단계    : {entry['phase']}")
        if entry.get("target_id"):
            print(f"  대상    : {entry['target_id']}")

        tool_log = entry.get("tool_calls_log", [])
        if tool_log:
            print(f"\n  [사용된 도구 — 총 {len(tool_log)}회]")
            for tc in tool_log:
                print(f"    - 도구명: {tc['name']}")
                args_str = json.dumps(tc["args"], ensure_ascii=False)
                print(f"      인자  : {args_str}")
        else:
            print(f"\n  [사용된 도구] 없음")

        print(f"\n  내용:\n")
        print(entry["content"])
        print()


def main():
    print("=" * 70)
    print(" Phase 1 입론 + Phase 2 연쇄논박 + Phase 3 자유논박 테스트")
    print("=" * 70)

    topic_dict = _load_topic(TOPIC_ID)

    personas = create_agents(
        topic=topic_dict,
        debate_format=DEBATE_FORMAT,
        user_stance=USER_STANCE,
        agent_intensities=AGENT_INTENSITIES,
    )

    snapshots = [
        AgentSnapshot(
            agent_id=p.agent_id,
            stance=p.stance,
            intensity=p.intensity,
            role_description=p.role_description,
            system_prompt=p.system_prompt,
            focus_area=p.focus_area,
        )
        for p in personas
    ]

    state = build_initial_state(
        topic=topic_dict["title"],
        user_stance=USER_STANCE,
        user_intensity=USER_INTENSITY,
        agents=snapshots,
        max_cycle=MAX_CYCLE,
    )

    print(f"\n  토픽: {topic_dict['title']}")
    print(f"  포맷: {DEBATE_FORMAT}, 사용자: {USER_STANCE}")
    print(f"  에이전트: {len(personas)}명")
    print(f"  최대 사이클: {MAX_CYCLE}\n")

    # ── 1단계: 입론 ──────────────────────────────────────────────────────────
    state = opening_arguments_node(state)

    # 사용자 입론 스킵 → phase를 chained_rebuttal로 전환
    state = dict(state)
    state["phase"] = "chained_rebuttal"

    # ── 2단계: 연쇄논박 ──────────────────────────────────────────────────────
    state = chained_rebuttal_node(state)

    # 사용자 연쇄논박 스킵 → phase를 free_rebuttal로 전환
    state = dict(state)
    state["phase"] = "free_rebuttal"

    # ── 3단계: 자유논박 ──────────────────────────────────────────────────────
    # 사용자 턴 스킵하며 max_cycle까지 반복
    cycle = 0
    while state["phase"] == "free_rebuttal" and cycle < MAX_CYCLE:
        state = free_rebuttal_node(state)
        state = dict(state)  # TypedDict → dict 변환 (수정 가능하도록)
        cycle += 1

    # ── 결과 출력 ────────────────────────────────────────────────────────────
    result_state = state

    for phase_name, phase_label in [
        ("opening", "입론"),
        ("chained_rebuttal", "연쇄논박"),
        ("free_rebuttal", "자유논박"),
    ]:
        entries = [e for e in result_state["debate_history"] if e["phase"] == phase_name]
        if not entries:
            continue
        print("=" * 70)
        print(f" {phase_label} 결과 ({len(entries)}건)")
        print("=" * 70)
        _print_entries(entries, phase_label)

    print("=" * 70)
    print(f" phase: {result_state['phase']}")
    print(f" current_cycle: {result_state['current_cycle']}/{result_state['max_cycle']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
