"""
interactive_test.py — 1~3단계 대화형 테스트

입론/연쇄논박은 더미 데이터로 빠르게 처리하고,
자유논박부터 직접 채팅하며 테스트할 수 있다.

터미널에서 실행:
    python src/stage3_free_rebuttal/interactive_test.py
"""

import json
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.phase0.persona_factory import create_agents
from src.state import AgentSnapshot, DebateEntry, build_initial_state
from src.stage1_opening.nodes import opening_arguments_node
from src.stage2_rebuttal.nodes import chained_rebuttal_node
from src.stage3_free_rebuttal.nodes import free_rebuttal_node


TOPIC_ID = "tech_002"
DEBATE_FORMAT = "2:2"
USER_STANCE = "PRO"
USER_INTENSITY = 3
AGENT_INTENSITIES = [3, 2, 4]

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "topics_20260323_processed.json"

USER_OPENING = """### 자기소개와 입장 표명
저는 사용자입니다. **환경 및 전력 규제 강화가 글로벌 데이터센터 산업의 최우선 과제**라고 생각합니다.

### 논거 1: 전력 소비의 지속 불가능성
데이터센터의 전력 소비는 이미 전 세계 전력 사용량의 1~2%를 차지하며, AI 확산으로 더 급증할 것입니다. **규제 없이 인프라만 확충하면 전력 위기가 가속화됩니다.**

### 논거 2: 환경 비용의 사회 전가
데이터센터 확장으로 인한 탄소 배출과 수자원 소비는 지역 사회에 전가되고 있습니다. **이익은 기업이 가져가고 비용은 사회가 부담하는 구조**는 규제로만 바꿀 수 있습니다.

### 결론
**환경 및 전력 규제 강화가 우선**이며, 이를 통해 지속 가능한 AI 발전의 기반을 마련해야 합니다."""

USER_REBUTTAL = """환경 및 전력 규제 없이 AI 인프라만 확충하면 **전력 위기와 환경 파괴가 가속화**됩니다. 기술 혁신으로 해결할 수 있다는 주장은 **아직 검증되지 않은 미래의 가능성**에 불과합니다. 따라서 **규제 강화가 우선**입니다."""


def _load_topic(topic_id: str) -> dict:
    with _DATA_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    for category_topics in data.get("categories", {}).values():
        for t in category_topics:
            if t["id"] == topic_id:
                return t
    raise ValueError(f"topic ID '{topic_id}'를 찾을 수 없습니다.")


def main():
    print("=" * 70)
    print(" 대화형 토론 테스트 (입론·연쇄논박 자동 → 자유논박 직접 채팅)")
    print("=" * 70)

    topic_dict = _load_topic(TOPIC_ID)
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

    print(f"\n  토픽: {topic_dict['title']}")
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
        phase="opening", content=USER_OPENING, target_id=None,
        tool_calls_log=[], json_raw="",
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
        print(f"  {entry['content'][:200]}...")
    print()

    # ── 2단계: 연쇄논박 (자동)
    print("-" * 70)
    print(" 2단계: 연쇄논박 (자동 처리)")
    print("-" * 70)
    state = chained_rebuttal_node(state)
    state = dict(state)
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
        phase="chained_rebuttal", content=USER_REBUTTAL, target_id="agent_2",
        tool_calls_log=[], json_raw="",
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
        print(f"    [{i+1}] {p.agent_id} — {p.role_description[:50]}")

    while True:
        choice = input(f"\n  상대 선택 (1~{len(opponents)}): ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(opponents):
            selected = opponents[int(choice) - 1]
            break
        print("  잘못된 입력입니다.")

    state["selected_opponent_id"] = selected.agent_id
    agent_map = {a["agent_id"]: a for a in state["agents"]}
    opponent_label = "찬성" if selected.stance == "PRO" else "반대"
    print(f"\n  ✓ 상대 선택: {selected.agent_id} ({opponent_label})")
    print(f"    역할: {selected.role_description[:80]}")

    # ── 자유논박 채팅 루프
    print("\n" + "=" * 70)
    print(f" 자유논박 시작: 사용자(찬성) ↔ {selected.agent_id}({opponent_label})")
    print(" 'q' 또는 'quit' 입력 시 종료")
    print("=" * 70)

    turn_count = 0
    while True:
        # 사용자 입력
        print(f"\n{'─' * 50}")
        user_input = input(f"  [사용자] 발언 (turn {turn_count * 2 + 1}): ").strip()
        if user_input.lower() in ("q", "quit", "exit"):
            print("\n  자유논박 종료.\n")
            break
        if not user_input:
            print("  빈 입력입니다. 다시 입력하세요.")
            continue

        # 사용자 발언 기록
        state["debate_history"].append(DebateEntry(
            turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
            phase="free_rebuttal", content=user_input, target_id=selected.agent_id,
            tool_calls_log=[], json_raw="",
        ))
        state["current_turn"] += 1

        # 에이전트 응답 생성
        state = free_rebuttal_node(state)
        state = dict(state)

        # 에이전트 최신 발언 출력
        agent_entries = [
            e for e in state["debate_history"]
            if e["speaker_id"] == selected.agent_id and e["phase"] == "free_rebuttal"
        ]
        if agent_entries:
            latest = agent_entries[-1]
            print(f"\n  [{selected.agent_id}] 발언 (turn {latest['turn']}):")
            print(f"  {latest['content']}")

        turn_count += 1

    # ── 결과 요약
    fr_entries = [e for e in state["debate_history"] if e["phase"] == "free_rebuttal"]
    print("=" * 70)
    print(f" 자유논박 결과 ({len(fr_entries)}건)")
    print("=" * 70)
    for entry in fr_entries:
        s_label = "찬성" if entry["stance"] == "PRO" else "반대"
        speaker = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]
        print(f"  [{speaker} ({s_label})] {entry['content'][:80]}...")
    print("=" * 70)


if __name__ == "__main__":
    main()
