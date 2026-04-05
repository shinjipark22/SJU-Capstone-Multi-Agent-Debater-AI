"""
test_run.py — 2단계 연쇄논박(Chained Rebuttal) 노드 독립 실행 테스트

사용자 입론/논박을 더미 데이터로 대체하여 전체 흐름을 테스트한다.

터미널에서 실행:
    python -m src.stage2_rebuttal.test_run
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


# ── 테스트 설정 ───────────────────────────────────────────────────────────────

TOPIC_ID = "tech_002"
DEBATE_FORMAT = "2:2"
USER_STANCE = "PRO"
USER_INTENSITY = 3
AGENT_INTENSITIES = [3, 2, 4]

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "topics_20260323_processed.json"

# ── 사용자 더미 데이터 ────────────────────────────────────────────────────────

USER_OPENING = """### 자기소개와 입장 표명
저는 사용자입니다. **환경 및 전력 규제 강화가 글로벌 데이터센터 산업의 최우선 과제**라고 생각합니다.

### 논거 1: 전력 소비의 지속 불가능성
데이터센터의 전력 소비는 이미 전 세계 전력 사용량의 1~2%를 차지하며, AI 확산으로 더 급증할 것입니다. **규제 없이 인프라만 확충하면 전력 위기가 가속화됩니다.**

### 논거 2: 환경 비용의 사회 전가
데이터센터 확장으로 인한 탄소 배출과 수자원 소비는 지역 사회에 전가되고 있습니다. **이익은 기업이 가져가고 비용은 사회가 부담하는 구조**는 규제로만 바꿀 수 있습니다.

### 결론
**환경 및 전력 규제 강화가 우선**이며, 이를 통해 지속 가능한 AI 발전의 기반을 마련해야 합니다."""


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
    print(" Phase 1 입론 + Phase 2 연쇄논박 테스트 (사용자 더미 포함)")
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
    )

    print(f"\n  토픽: {topic_dict['title']}")
    print(f"  포맷: {DEBATE_FORMAT}, 사용자: {USER_STANCE}")
    print(f"  에이전트: {len(personas)}명\n")

    # ── 1단계: 입론 ──────────────────────────────────────────────────────────
    state = opening_arguments_node(state)

    # 사용자 입론 더미 삽입
    state = dict(state)
    user_turn = len([e for e in state["debate_history"] if e["phase"] == "opening"])
    user_entry = DebateEntry(
        turn=user_turn,
        speaker_id="user",
        stance=USER_STANCE,
        phase="opening",
        content=USER_OPENING,
        target_id=None,
        tool_calls_log=[],
        json_raw="",
    )
    state["debate_history"].append(user_entry)
    state["debate_history"].sort(key=lambda e: e["turn"])
    state["phase"] = "chained_rebuttal"
    print(f"  [사용자] 입론 더미 삽입 완료 (turn={user_turn})\n")

    # ── 2단계: 연쇄논박 ──────────────────────────────────────────────────────
    result_state = chained_rebuttal_node(state)

    # ── 결과 출력 ────────────────────────────────────────────────────────────
    for phase_name in ("opening", "chained_rebuttal"):
        phase_label = "입론" if phase_name == "opening" else "연쇄논박"
        print("=" * 70)
        print(f" {phase_label} 결과")
        print("=" * 70)

        for entry in result_state["debate_history"]:
            if entry["phase"] != phase_name:
                continue

            stance = entry["stance"]
            s_label = "찬성" if stance == "PRO" else "반대"
            speaker = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]

            print("-" * 70)
            print(f"  발언자  : {speaker} ({s_label})")
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

    print("=" * 70)
    print(f" phase: {result_state['phase']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
