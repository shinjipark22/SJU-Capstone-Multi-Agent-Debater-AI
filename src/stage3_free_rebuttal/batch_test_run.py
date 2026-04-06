"""
batch_test_run.py — 1~3단계 전체 테스트 → txt 파일 출력

터미널에서 실행:
    python src/stage3_free_rebuttal/batch_test_run.py
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


TOPIC_ID = "tech_002"
DEBATE_FORMAT = "2:2"
USER_STANCE = "PRO"
USER_INTENSITY = 3
AGENT_INTENSITIES = [3, 2, 4]

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "topics_20260323_processed.json"
_OUTPUT_DIR = Path(__file__).parent.parent.parent / "test_results"

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


def _format_entry(entry: dict) -> str:
    s_label = "찬성" if entry["stance"] == "PRO" else "반대"
    speaker = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]
    target = entry.get("target_id", "")
    target_str = f" → {target}" if target else ""
    lines = []
    lines.append(f"[턴 {entry['turn']}] {speaker} ({s_label}){target_str}")
    lines.append(entry["content"])
    return "\n".join(lines)


def main():
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

    # ── 3단계: 자유논박 (1:1 핑퐁 — 사용자 vs agent_2)
    # 사용자가 상대 에이전트를 선택 (테스트에서는 CON 에이전트 중 하나)
    con_agents = [a.agent_id for a in personas if a.stance == "CON"]
    selected_opponent = con_agents[0] if con_agents else personas[0].agent_id
    state["selected_opponent_id"] = selected_opponent
    print(f"\n  [자유논박] 상대 선택: {selected_opponent}\n")

    USER_FREE_REBUTTALS = [
        "기술 혁신이 환경 문제를 해결할 수 있다는 주장은 아직 검증되지 않았습니다. 실제로 데이터센터의 탄소 배출은 매년 증가하고 있습니다. 규제 없이 이 추세를 어떻게 바꿀 수 있다고 보십니까?",
        "기업의 자발적 노력만으로는 한계가 있습니다. 유럽연합은 이미 데이터센터 에너지 효율 기준을 도입하여 성과를 거두고 있습니다. 이것이 혁신을 억누르고 있다고 볼 수 있습니까?",
        "AI 인프라 확충이 경제 성장에 기여한다는 점은 인정합니다. 하지만 그 과정에서 발생하는 환경 비용은 누가 부담해야 합니까?",
    ]

    # 3턴씩 핑퐁 (사용자 → 에이전트 → 사용자 → 에이전트 → ...)
    for i, user_msg in enumerate(USER_FREE_REBUTTALS):
        # 사용자 발언 삽입
        state["debate_history"].append(DebateEntry(
            turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
            phase="free_rebuttal", content=user_msg, target_id=selected_opponent,
            tool_calls_log=[], json_raw="",
        ))
        state["current_turn"] += 1
        print(f"  [사용자] 자유논박 발언 {i+1} 삽입")

        # 에이전트 응답 생성
        state = free_rebuttal_node(state)
        state = dict(state)

    # ── 결과 txt 파일 생성
    _OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = _OUTPUT_DIR / f"debate_{timestamp}.txt"

    lines = []
    lines.append(f"토픽: {topic_dict['title']}")
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
    lines.append("=" * 70)

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n결과 저장: {output_path}")
    print(f"총 {len(state['debate_history'])}건 발언")


if __name__ == "__main__":
    main()
