"""
interactive_test.py — 1~3단계 완전 대화형 테스트

모든 단계에서 사용자가 직접 발언을 입력한다.
AI 에이전트 발언은 자동 생성.
종료 시 전체 결과 txt 저장.

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


def _get_multiline_input(prompt: str) -> str:
    """여러 줄 입력 받기. 빈 줄 2번 연속 입력하면 종료."""
    print(prompt)
    print("  (입력 후 빈 줄 2번 연속으로 Enter 치면 완료)")
    lines = []
    empty_count = 0
    while True:
        line = input("  ")
        if line == "":
            empty_count += 1
            if empty_count >= 2:
                break
            lines.append("")
        else:
            empty_count = 0
            lines.append(line)
    return "\n".join(lines).strip()


def main():
    topics = _load_all_topics()
    print("=" * 70)
    print(" 1~3단계 완전 대화형 테스트")
    print(" 모든 단계에서 사용자가 직접 발언합니다.")
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

    print(f"\n  ✓ 선택: {topic_dict['title']}")
    print(f"  사용자 입장: 찬성(PRO)\n")

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
        topic_id=topic_dict["id"],
    )

    # ══════════════════════════════════════════════════════
    # 1단계: 입론
    # ══════════════════════════════════════════════════════
    print("=" * 70)
    print(" 1단계: 입론")
    print("=" * 70)
    print("\n  AI 에이전트 입론 생성 중...\n")
    state = opening_arguments_node(state)
    state = dict(state)

    # AI 입론 결과 출력
    for entry in state["debate_history"]:
        if entry["phase"] != "opening":
            continue
        s_label = "찬성" if entry["stance"] == "PRO" else "반대"
        speaker = entry["speaker_id"]
        print(f"  [{speaker} ({s_label})]")
        print(f"  {entry['content']}")
        print()

    # 사용자 입론 입력
    print("-" * 70)
    user_opening = _get_multiline_input(f"  [사용자 입론] 찬성 입장에서 입론을 작성하세요:")
    if not user_opening:
        print("  입론이 비어있습니다. 종료합니다.")
        return

    user_turn = len([e for e in state["debate_history"] if e["phase"] == "opening"])
    state["debate_history"].append(DebateEntry(
        turn=user_turn, speaker_id="user", stance=USER_STANCE,
        phase="opening", content=user_opening,
        target_id=None, tool_calls_log=[], json_raw="",
    ))
    state["debate_history"].sort(key=lambda e: e["turn"])
    state["phase"] = "chained_rebuttal"
    print(f"\n  ✓ 사용자 입론 등록 완료\n")

    # ══════════════════════════════════════════════════════
    # 2단계: 연쇄논박
    # ══════════════════════════════════════════════════════
    print("=" * 70)
    print(" 2단계: 연쇄논박")
    print("=" * 70)
    print("\n  AI 에이전트 연쇄논박 생성 중...\n")
    state = chained_rebuttal_node(state)
    state = dict(state)

    # AI 연쇄논박 결과 출력
    for entry in state["debate_history"]:
        if entry["phase"] != "chained_rebuttal":
            continue
        s_label = "찬성" if entry["stance"] == "PRO" else "반대"
        speaker = entry["speaker_id"]
        target = entry.get("target_id", "")
        print(f"  [{speaker} ({s_label}) → {target}]")
        print(f"  {entry['content']}")
        print()

    # 사용자를 공격한 상대 발언 찾기
    attacker_to_user = None
    for e in reversed(state["debate_history"]):
        if e["phase"] == "chained_rebuttal" and e["target_id"] == "user":
            attacker_to_user = e
            break

    if attacker_to_user:
        print("-" * 70)
        print(f"  ↑ {attacker_to_user['speaker_id']}가 사용자를 공격했습니다.")
        print(f"  이에 대해 반박하세요.\n")
        rebuttal_target = attacker_to_user["speaker_id"]
    else:
        print("-" * 70)
        print("  반박할 상대를 선택하세요.")
        con_agents = [a["agent_id"] for a in state["agents"] if a["stance"] == "CON"]
        rebuttal_target = con_agents[0] if con_agents else "agent_1"

    user_rebuttal = _get_multiline_input(f"  [사용자 연쇄논박] → {rebuttal_target} 반박:")
    if not user_rebuttal:
        print("  연쇄논박이 비어있습니다. 종료합니다.")
        return

    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
        phase="chained_rebuttal", content=user_rebuttal,
        target_id=rebuttal_target, tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "free_rebuttal"
    print(f"\n  ✓ 사용자 연쇄논박 등록 완료\n")

    # ══════════════════════════════════════════════════════
    # 3단계: 자유논박
    # ══════════════════════════════════════════════════════
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

    print("\n" + "=" * 70)
    print(f" 자유논박: 사용자(찬성) ↔ {selected.agent_id}({opponent_label})")
    print(f" 토픽: {topic_dict['title']}")
    print(" 'q' 입력 시 종료 → 결과 txt 저장")
    print("=" * 70)

    turn_count = 0
    while True:
        print(f"\n{'─' * 50}")
        user_input = input(f"  [사용자] 발언: ").strip()
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
            print(f"\n  [{selected.agent_id}] 발언:")
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
