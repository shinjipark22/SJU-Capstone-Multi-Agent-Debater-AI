"""
interactive_test.py — 1~3단계 전체 대화형 테스트

모든 토픽에서 선택 가능. 입론/연쇄논박은 자동 처리 후,
자유논박부터 직접 채팅. 종료 시 전체 결과 txt 저장.

터미널에서 실행:
    python src/stage3_free_rebuttal/interactive_test.py
"""

import json
import sys
import os
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.phase0.persona_factory import create_agents
from src.state import AgentSnapshot, DebateEntry, build_initial_state
from src.stage1_opening.nodes import opening_arguments_node
from src.stage2_rebuttal.nodes import chained_rebuttal_node
from src.stage3_free_rebuttal.nodes import free_rebuttal_node


DEBATE_FORMAT = "2:2"
USER_STANCE = "PRO"
USER_INTENSITY = 3
AGENT_INTENSITIES = [3, 2, 4]

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "topics_20260323_processed.json"
_OUTPUT_DIR = Path(__file__).parent.parent.parent / "test_results"


def _load_all_topics() -> list:
    with _DATA_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    topics = []
    for category_topics in data.get("categories", {}).values():
        for t in category_topics:
            topics.append(t)
    return topics


def _make_user_opening(topic_title: str) -> str:
    return f"""### 자기소개와 입장 표명
저는 사용자입니다. **'{topic_title}'에 찬성**합니다.

### 논거 1
이 주제에서 제시된 방향은 현재 상황의 근본적 문제를 해결하는 데 더 효과적입니다. 기존 접근 방식만으로는 문제가 악화될 수 있으며, 새로운 방향의 전환이 시급합니다.

### 논거 2
반대 측이 주장하는 대안은 단기적 이익에 치중되어 있으며, 장기적으로 발생할 수 있는 부작용을 충분히 고려하지 않고 있습니다. 지속 가능한 해결을 위해서는 찬성 측의 접근이 필요합니다.

### 결론
따라서 **'{topic_title}'에 찬성**하며, 이를 통해 더 나은 미래를 만들 수 있다고 확신합니다."""


def _make_user_rebuttal(topic_title: str) -> str:
    return f"""반대 측의 주장은 현실적 근거가 부족하며, '{topic_title}'의 핵심 논점을 회피하고 있습니다. 기존 방식의 한계는 이미 여러 사례에서 입증되었으며, 새로운 접근이 필요한 시점입니다."""


def _format_entry(entry: dict) -> str:
    s_label = "찬성" if entry["stance"] == "PRO" else "반대"
    speaker = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]
    target = entry.get("target_id", "")
    target_str = f" → {target}" if target else ""
    return f"[턴 {entry['turn']}] {speaker} ({s_label}){target_str}\n{entry['content']}"


def _save_results(state: dict, topic_dict: dict, personas: list):
    _OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    topic_id = topic_dict["id"]
    output_path = _OUTPUT_DIR / f"debate_{topic_id}_{timestamp}.txt"

    lines = []
    lines.append(f"토픽: {topic_dict['title']}")
    lines.append(f"토픽 ID: {topic_id}")
    lines.append(f"포맷: {DEBATE_FORMAT} | 사용자: {USER_STANCE}")
    lines.append(f"에이전트: {len(personas)}명")
    lines.append(f"생성 시각: {timestamp}")
    lines.append("")

    for phase_name, phase_label in [
        ("opening", "1단계: 입론"),
        ("chained_rebuttal", "2단계: 연쇄논박"),
        ("free_rebuttal", "3단계: 자유논박"),
    ]:
        entries = [e for e in state["debate_history"] if e["phase"] == phase_name]
        lines.append("=" * 70)
        lines.append(f" {phase_label} ({len(entries)}건)")
        lines.append("=" * 70)
        for entry in entries:
            lines.append("-" * 70)
            lines.append(_format_entry(entry))
            lines.append("")

    lines.append("=" * 70)
    lines.append(f"최종 phase: {state['phase']}")
    lines.append(f"총 발언: {len(state['debate_history'])}건")
    lines.append("=" * 70)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n  결과 저장: {output_path}")
    print(f"  총 {len(state['debate_history'])}건 발언")


def main():
    topics = _load_all_topics()
    print("=" * 70)
    print(" 1~3단계 전체 대화형 테스트")
    print("=" * 70)
    print(f"\n  토픽 목록 ({len(topics)}개):\n")
    for i, t in enumerate(topics):
        print(f"    [{i+1:2d}] {t['id']:10s} | {t['title']}")

    while True:
        choice = input(f"\n  토픽 선택 (1~{len(topics)}, q=종료): ").strip()
        if choice.lower() in ("q", "quit"):
            return
        if choice.isdigit() and 1 <= int(choice) <= len(topics):
            topic_dict = topics[int(choice) - 1]
            break
        print("  잘못된 입력입니다.")

    print(f"\n  ✓ 선택: {topic_dict['title']}\n")

    # ── 에이전트 생성
    personas = create_agents(
        topic=topic_dict, debate_format=DEBATE_FORMAT,
        user_stance=USER_STANCE, agent_intensities=AGENT_INTENSITIES,
    )
    snapshots = [
        AgentSnapshot(
            agent_id=p.agent_id, stance=p.stance, intensity=p.intensity,
            role_description=p.role_description, system_prompt=p.system_prompt,
            focus_area=p.focus_area,
        ) for p in personas
    ]
    state = build_initial_state(
        topic=topic_dict["title"], user_stance=USER_STANCE,
        user_intensity=USER_INTENSITY, agents=snapshots,
    )

    print(f"  토픽: {topic_dict['title']}")
    print(f"  포맷: {DEBATE_FORMAT}, 사용자: {USER_STANCE}")
    print(f"  에이전트: {len(personas)}명\n")

    # ── 1단계: 입론 (자동)
    print("-" * 70)
    print(" 1단계: 입론 (자동 처리)")
    print("-" * 70)
    state = opening_arguments_node(state)
    state = dict(state)
    user_turn = len([e for e in state["debate_history"] if e["phase"] == "opening"])
    state["debate_history"].append(DebateEntry(
        turn=user_turn, speaker_id="user", stance=USER_STANCE,
        phase="opening", content=_make_user_opening(topic_dict["title"]),
        target_id=None, tool_calls_log=[], json_raw="",
    ))
    state["debate_history"].sort(key=lambda e: e["turn"])
    state["phase"] = "chained_rebuttal"
    print("  ✓ 입론 완료\n")

    # 입론 결과 출력
    print("=" * 70)
    print(" 입론 결과")
    print("=" * 70)
    for entry in state["debate_history"]:
        if entry["phase"] != "opening":
            continue
        s_label = "찬성" if entry["stance"] == "PRO" else "반대"
        speaker = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]
        print(f"\n  [{speaker} ({s_label})]")
        print(f"  {entry['content'][:300]}...")
    print()

    # ── 2단계: 연쇄논박 (자동)
    print("-" * 70)
    print(" 2단계: 연쇄논박 (자동 처리)")
    print("-" * 70)
    state = chained_rebuttal_node(state)
    state = dict(state)
    con_agents = [a["agent_id"] for a in state["agents"] if a["stance"] == "CON"]
    rebuttal_target = con_agents[0] if con_agents else "agent_1"
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
        phase="chained_rebuttal", content=_make_user_rebuttal(topic_dict["title"]),
        target_id=rebuttal_target, tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "free_rebuttal"
    print("  ✓ 연쇄논박 완료\n")

    # 연쇄논박 결과 출력
    print("=" * 70)
    print(" 연쇄논박 결과")
    print("=" * 70)
    for entry in state["debate_history"]:
        if entry["phase"] != "chained_rebuttal":
            continue
        s_label = "찬성" if entry["stance"] == "PRO" else "반대"
        speaker = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]
        target = entry.get("target_id", "")
        print(f"\n  [{speaker} ({s_label}) → {target}]")
        print(f"  {entry['content'][:300]}")
    print()

    # ── 상대 에이전트 선택
    print("=" * 70)
    print(" 3단계: 자유논박 — 상대 에이전트 선택")
    print("=" * 70)
    opposite = "CON" if USER_STANCE == "PRO" else "PRO"
    opponents = [p for p in personas if p.stance == opposite]
    print(f"\n  상대팀({opposite}) 에이전트 목록:")
    for i, p in enumerate(opponents):
        print(f"    [{i+1}] {p.agent_id} — {p.role_description[:60]}")

    while True:
        choice = input(f"\n  상대 선택 (1~{len(opponents)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(opponents):
            selected = opponents[int(choice) - 1]
            break
        print("  잘못된 입력입니다.")

    state["selected_opponent_id"] = selected.agent_id
    opponent_label = "찬성" if selected.stance == "PRO" else "반대"
    print(f"\n  ✓ 상대 선택: {selected.agent_id} ({opponent_label})")
    print(f"    역할: {selected.role_description[:80]}")

    # ── 자유논박 채팅 루프
    print("\n" + "=" * 70)
    print(f" 자유논박 시작: 사용자(찬성) ↔ {selected.agent_id}({opponent_label})")
    print(f" 토픽: {topic_dict['title']}")
    print(" 'q' 또는 'quit' 입력 시 종료 → 결과 txt 저장")
    print("=" * 70)

    turn_count = 0
    while True:
        print(f"\n{'─' * 50}")
        user_input = input(f"  [사용자] 발언 (turn {turn_count * 2 + 1}): ").strip()
        if user_input.lower() in ("q", "quit", "exit"):
            print("\n  자유논박 종료.\n")
            break
        if not user_input:
            print("  빈 입력입니다. 다시 입력하세요.")
            continue

        state["debate_history"].append(DebateEntry(
            turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
            phase="free_rebuttal", content=user_input, target_id=selected.agent_id,
            tool_calls_log=[], json_raw="",
        ))
        state["current_turn"] += 1

        state = free_rebuttal_node(state)
        state = dict(state)

        agent_entries = [
            e for e in state["debate_history"]
            if e["speaker_id"] == selected.agent_id and e["phase"] == "free_rebuttal"
        ]
        if agent_entries:
            latest = agent_entries[-1]
            print(f"\n  [{selected.agent_id}] 발언 (turn {latest['turn']}):")
            print(f"  {latest['content']}")

        turn_count += 1

    # ── 결과 저장
    _save_results(state, topic_dict, personas)

    # ── 다른 토픽 계속?
    again = input("\n  다른 토픽으로 계속하시겠습니까? (y/n): ").strip()
    if again.lower() == 'y':
        main()


if __name__ == "__main__":
    main()
