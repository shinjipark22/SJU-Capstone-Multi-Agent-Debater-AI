"""
test_run.py — 5단계 종합 및 재개념화 테스트 (1~5단계 포함)

터미널에서 실행:
    python src/stage5_synthesis/test_run.py
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
from src.stage5_synthesis.nodes import synthesis_node


TOPIC_ID = "tech_001"
DEBATE_FORMAT = "2:2"
USER_STANCE = "PRO"
USER_INTENSITY = 3
AGENT_INTENSITIES = [3, 2, 4]

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "topics_20260323_processed.json"

USER_OPENING = """### 자기소개와 입장 표명
저는 사용자입니다. **AI가 인간의 일자리를 대체하는 것은 불가피한 흐름**이라고 생각합니다.

### 논거 1: 기술 발전의 역사적 패턴
산업혁명 이후 모든 기술 혁신은 기존 일자리를 대체하면서 새로운 일자리를 창출해왔습니다. **AI 역시 같은 패턴을 따를 것**이며, 직업 구조의 개편은 자연스러운 현상입니다.

### 논거 2: 생산성 향상과 경제 성장
AI 도입으로 반복적이고 위험한 업무가 자동화되면 **인간은 더 창의적이고 가치 있는 일에 집중**할 수 있습니다. 이는 전체 경제의 생산성을 높이는 긍정적 변화입니다.

### 결론
**AI에 의한 직업 구조 변화는 막을 수 없으며**, 이를 두려워하기보다 적극적으로 대비해야 합니다."""

USER_REBUTTAL = """AI로 인한 일자리 변화가 불가피하다는 점은 인정하지만, **전환 과정에서 발생하는 실업과 불평등을 방치해서는 안 됩니다.** 기술 발전의 혜택이 모두에게 돌아가도록 **정책적 개입이 필수적**입니다."""

USER_FREE_REBUTTAL = """단순히 시장에 맡기면 **기술 격차가 소득 격차로 이어질 것**입니다. AI 전환기에 정부의 적극적인 재교육 정책과 사회 안전망 강화가 반드시 필요합니다."""

USER_ROLE_REVERSAL = """### 논거 1: 당장의 실업 문제
AI 자동화로 인해 단기적으로 대규모 실업이 발생할 수 있으며, **재취업까지의 공백 기간에 생계 위협**을 받는 노동자가 속출할 수 있습니다.

### 논거 2: 기술 격차 심화
AI 활용 능력의 차이가 **새로운 형태의 불평등**을 만들 수 있습니다. 디지털 리터러시가 부족한 계층은 구조적으로 배제될 위험이 있습니다.

### 결론
AI 발전의 부작용을 최소화하기 위한 **사회적 안전장치 마련이 시급**합니다."""

USER_SYNTHESIS = """### 찬성측 타당한 점
AI에 의한 직업 구조 개편은 역사적 패턴상 불가피하며, 생산성 향상과 새로운 일자리 창출이라는 긍정적 측면이 있습니다.

### 반대측 타당한 점
그러나 전환 과정에서 발생하는 대규모 실업과 기술 격차로 인한 불평등 심화는 심각한 사회 문제입니다.

### 최적해
**AI 직업 전환 3단계 정책**을 제안합니다. 첫째, 정부는 **AI 전환 영향 평가 제도**를 도입하여 기업이 AI 도입 시 고용 영향을 사전 보고하도록 의무화해야 합니다. 둘째, **AI 재교육 바우처 제도**를 시행하여 실직 위험 노동자에게 6개월~1년간 무상 직업 훈련과 생활비를 지원해야 합니다. 셋째, **AI 활용 기업에 대한 고용 전환 세액공제**를 통해 기존 직원을 해고 대신 재배치하는 기업에 인센티브를 제공해야 합니다."""


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
    print(" Phase 1~5 테스트 (입론 + 연쇄논박 + 자유논박 + 역할반전 + 종합)")
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

    # ── 3단계: 자유논박 (1사이클)
    opponent_agents = [a for a in state["agents"] if a["stance"] != USER_STANCE]
    if opponent_agents:
        state["selected_opponent_id"] = opponent_agents[0]["agent_id"]
    state = free_rebuttal_node(state)
    state = dict(state)
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user", stance=USER_STANCE,
        phase="free_rebuttal", content=USER_FREE_REBUTTAL,
        target_id=state.get("selected_opponent_id"),
        tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "role_reversal"
    print(f"  [사용자] 자유논박 더미 삽입 완료\n")

    # ── 4단계: 역할반전
    state = role_reversal_node(state)
    state = dict(state)
    reversed_user_stance = "CON" if USER_STANCE == "PRO" else "PRO"
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user",
        stance=reversed_user_stance, phase="role_reversal",
        content=USER_ROLE_REVERSAL, target_id=None,
        tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "synthesis"
    print(f"  [사용자] 역할반전 더미 삽입 완료\n")

    # ── 5단계: 종합 및 재개념화
    state = synthesis_node(state)
    state = dict(state)
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user",
        stance=USER_STANCE, phase="synthesis",
        content=USER_SYNTHESIS, target_id=None,
        tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["is_finished"] = True
    print(f"  [사용자] 종합 더미 삽입 완료\n")

    # ── 종합 결과 출력
    entries = [e for e in state["debate_history"] if e["phase"] == "synthesis"]
    print("=" * 70)
    print(f" 종합 및 재개념화 결과 ({len(entries)}건)")
    print("=" * 70)
    for entry in entries:
        s_label = "찬성" if entry["stance"] == "PRO" else "반대"
        speaker = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]
        print("-" * 70)
        print(f"  {speaker} (원래 {s_label}) | 턴 {entry['turn']}")
        print(f"\n{entry['content']}\n")

    print("=" * 70)
    print(f" phase: {state['phase']}, is_finished: {state['is_finished']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
