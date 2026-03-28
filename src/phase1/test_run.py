"""
test_run.py — Phase 1 입론 노드 독립 실행 테스트

터미널에서 직접 실행:
    python -m src.phase1.test_run
    (또는 프로젝트 루트에서 PYTHONPATH=. python src/phase1/test_run.py)

[테스트 시나리오]
    - data/topics_20260323_processed.json 에서 TOPIC_ID로 주제 로드
    - 토론 포맷 2:2 (AI 3명: CON 2명 + PRO 1명, 사용자 PRO)
    - 각 AI 에이전트가 입론을 순서대로 생성하는지 확인
    - debate_history 누적 및 phase 전환 검증
"""

import json
import sys
import os
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가 (직접 실행 시 대비)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.phase0.persona_factory import create_agents
from src.state import AgentSnapshot, build_initial_state
from src.phase1.nodes import opening_arguments_node


# ── 테스트 설정 ───────────────────────────────────────────────────────────────

TOPIC_ID = "tech_001"          # data/topics_20260323_processed.json 에서 사용할 토픽 ID
DEBATE_FORMAT = "2:2"
USER_STANCE = "PRO"
USER_INTENSITY = 3
AGENT_INTENSITIES = [4, 2, 5]  # CON(강경), CON(온건), PRO(매우강경) — 2:2 포맷 AI 3명

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "topics_20260323_processed.json"


def _load_topic(topic_id: str) -> dict:
    """topics_20260323_processed.json에서 topic_id에 해당하는 항목을 반환한다."""
    if not _DATA_PATH.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {_DATA_PATH}")

    with _DATA_PATH.open(encoding="utf-8") as f:
        data = json.load(f)

    for category_topics in data.get("categories", {}).values():
        for t in category_topics:
            if t["id"] == topic_id:
                return t

    raise ValueError(f"topic ID '{topic_id}'를 찾을 수 없습니다.")


def build_test_state():
    """테스트용 초기 DebateState를 생성한다."""
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
        )
        for p in personas
    ]

    state = build_initial_state(
        topic=topic_dict["title"],
        user_stance=USER_STANCE,
        user_intensity=USER_INTENSITY,
        agents=snapshots,
    )

    return state, personas, topic_dict


def print_separator(char: str = "─", width: int = 70) -> None:
    print(char * width)


def main():
    print_separator("═")
    print(" Phase 1 입론(Opening Arguments) 노드 테스트")
    print_separator("═")

    # ── 초기 State 구성 ───────────────────────────────────────────────────────
    print("\n[1] 초기 State 구성 중...")
    state, personas, topic_dict = build_test_state()

    print(f"  토픽 ID    : {topic_dict['id']}")
    print(f"  토론 주제  : {state['topic']}")
    print(f"  사용자 진영: {state['user_stance']}")
    print(f"  발언 순서  : {state['speaking_order']}")
    print(f"  현재 단계  : {state['phase']}")
    print(f"  에이전트 수: {len(state['agents'])}명")
    print()
    for p in personas:
        print(f"    [{p.agent_id}] {p.stance} | 강경도 {p.intensity} | {p.role_description}")

    # ── 노드 실행 ─────────────────────────────────────────────────────────────
    print_separator()
    print("\n[2] opening_arguments_node 실행\n")
    print_separator()

    result_state = opening_arguments_node(state)

    # ── 결과 검증 ─────────────────────────────────────────────────────────────
    print_separator()
    print("\n[3] 결과 검증\n")

    history = result_state["debate_history"]
    ai_count = len([sid for sid in state["speaking_order"] if sid != "user"])

    # debate_history 항목 수 검증
    assert len(history) == ai_count, (
        f"❌ debate_history 길이 불일치: 기대 {ai_count}, 실제 {len(history)}"
    )
    print(f"  ✅ debate_history 길이: {len(history)}개 (AI 에이전트 수와 일치)")

    # phase 전환 검증
    assert result_state["phase"] == "chained_rebuttal", (
        f"❌ phase 전환 실패: 기대 'chained_rebuttal', 실제 '{result_state['phase']}'"
    )
    print(f"  ✅ phase 전환: '{state['phase']}' → '{result_state['phase']}'")

    # current_turn 증가 검증
    assert result_state["current_turn"] == ai_count, (
        f"❌ current_turn 불일치: 기대 {ai_count}, 실제 {result_state['current_turn']}"
    )
    print(f"  ✅ current_turn: {result_state['current_turn']} (발언 횟수와 일치)")

    # ── 입론 내용 출력 ─────────────────────────────────────────────────────────
    print_separator()
    print("\n[4] 생성된 입론 내용\n")

    for entry in history:
        print_separator("-")
        print(f"  발언자  : {entry['speaker_id']} ({entry['stance']})")
        print(f"  턴 번호 : {entry['turn']}")
        print(f"  단계    : {entry['phase']}")
        print(f"  입론 내용:\n")
        # 긴 텍스트는 80자 단위로 출력
        content = entry["content"]
        for i in range(0, len(content), 80):
            print(f"    {content[i:i+80]}")
        print()

    print_separator("═")
    print(" ✅ Phase 1 테스트 완료")
    print_separator("═")


if __name__ == "__main__":
    main()
