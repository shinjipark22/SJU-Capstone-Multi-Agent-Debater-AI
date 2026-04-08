"""
test_run.py — 4단계 역할반전 테스트 (1~3단계 포함)

터미널에서 실행:
    python src/stage4_role_reversal/test_run.py
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
from src.stage4_role_reversal.nodes import role_reversal_node


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

USER_FREE_REBUTTAL = """상대의 주장은 기술 혁신이 규제를 대체할 수 있다는 전제에 기반하지만, **현실에서 자율 규제만으로는 환경 문제가 해결된 사례가 없습니다.** 따라서 규제 강화가 먼저 이루어져야 합니다."""

USER_ROLE_REVERSAL = """### 논거 1: 기술 혁신의 잠재력
규제보다 기술 혁신이 데이터센터의 환경 문제를 더 효과적으로 해결할 수 있습니다. **액침냉각, 재생에너지 전환 등 기술 발전이 빠르게 이루어지고 있으며**, 과도한 규제는 이러한 혁신을 저해할 수 있습니다.

### 논거 2: 경제적 경쟁력 유지
엄격한 환경 규제는 데이터센터 산업의 글로벌 경쟁력을 약화시킬 수 있습니다. **규제가 약한 국가로 산업이 이전되면 오히려 전체 탄소 배출이 증가**하는 역효과가 발생합니다.

### 결론
**기술 혁신과 시장 원리를 통한 자율적 개선**이 규제보다 효과적일 수 있으며, 균형 잡힌 접근이 필요합니다."""


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
    print(" Phase 1~4 테스트 (입론 + 연쇄논박 + 자유논박 + 역할반전)")
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
        topic_id=topic_dict["id"],
    )

    print(f"\n  토픽: {topic_dict['title']}")
    print(f"  포맷: {DEBATE_FORMAT}, 사용자: {USER_STANCE}")
    print(f"  에이전트: {len(personas)}명\n")

    # ── 1단계: 입론
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
    print(f"  [사용자] 입론 더미 삽입 완료\n")

    # ── 2단계: 연쇄논박
    state = chained_rebuttal_node(state)
    state = dict(state)
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
        phase="chained_rebuttal", content=USER_REBUTTAL, target_id="agent_2",
        tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "free_rebuttal"
    print(f"  [사용자] 연쇄논박 더미 삽입 완료\n")

    # ── 3단계: 자유논박 (1사이클만)
    opponent_agents = [a for a in state["agents"] if a["stance"] != USER_STANCE]
    if opponent_agents:
        state["selected_opponent_id"] = opponent_agents[0]["agent_id"]
    state = free_rebuttal_node(state)
    state = dict(state)
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
        phase="free_rebuttal", content=USER_FREE_REBUTTAL, target_id=state.get("selected_opponent_id"),
        tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "role_reversal"
    print(f"  [사용자] 자유논박 더미 삽입 완료\n")

    # ── 4단계: 역할반전
    state = role_reversal_node(state)
    state = dict(state)

    # 사용자 역할반전 더미 삽입
    reversed_user_stance = "CON" if USER_STANCE == "PRO" else "PRO"
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user",
        stance=reversed_user_stance,  # 반전된 입장
        phase="role_reversal", content=USER_ROLE_REVERSAL, target_id=None,
        tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    print(f"  [사용자] 역할반전 더미 삽입 완료\n")

    # ── 역할반전 결과 출력
    entries = [e for e in state["debate_history"] if e["phase"] == "role_reversal"]
    print("=" * 70)
    print(f" 역할반전 결과 ({len(entries)}건)")
    print("=" * 70)
    for entry in entries:
        s_label = "찬성" if entry["stance"] == "PRO" else "반대"
        speaker = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]
        print("-" * 70)
        print(f"  {speaker} → {s_label} 옹호 (역할반전) | 턴 {entry['turn']}")
        print(f"\n{entry['content']}\n")

    print("=" * 70)
    print(f" phase: {state['phase']}, role_reversed: {state['role_reversed']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
