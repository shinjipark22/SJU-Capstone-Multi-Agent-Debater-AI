"""
auto_test_all_topics.py — 12개 토픽 자동 테스트 (1~5단계 전체)

사용자 입력은 tests/user_inputs_all_topics.json에서 로드.
결과는 test_results/auto_test_{topic_id}_{timestamp}.txt로 저장.

실행:
    python tests/auto_test_all_topics.py
    python tests/auto_test_all_topics.py --topic tech_001  # 특정 토픽만
"""

import json
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.phase0.persona_factory import create_agents
from src.state import AgentSnapshot, DebateEntry, build_initial_state
from src.stage1_opening.nodes import opening_arguments_node
from src.stage2_rebuttal.nodes import chained_rebuttal_node
from src.stage3_free_rebuttal.nodes import free_rebuttal_node
from src.stage4_role_reversal.nodes import role_reversal_node
from src.stage5_synthesis.nodes import synthesis_node, synthesis_discuss_node

_DATA_PATH = Path(__file__).parent.parent / "data" / "topics_20260323_processed.json"
_USER_INPUTS_PATH = Path(__file__).parent / "user_inputs_all_topics.json"
_OUTPUT_DIR = Path(__file__).parent.parent / "test_results"


def _load_topic(topic_id: str) -> dict:
    with _DATA_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    for category_topics in data.get("categories", {}).values():
        for t in category_topics:
            if t["id"] == topic_id:
                return t
    raise ValueError(f"topic ID '{topic_id}'를 찾을 수 없습니다.")


def _save_results(topic_id: str, topic_title: str, state: dict, user_inputs: dict, config: dict):
    _OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fmt = config["debate_format"].replace(":", "v")
    output_path = _OUTPUT_DIR / f"auto_test_{fmt}_{topic_id}_{timestamp}.txt"

    lines = [
        f"{'='*70}",
        f" 자동 테스트 결과 — {topic_id}",
        f"{'='*70}",
        f"토픽: {topic_title}",
        f"토픽 ID: {topic_id}",
        f"포맷: {config['debate_format']} | 사용자: {config['user_stance']}",
        f"생성 시각: {timestamp}",
        "",
    ]

    for phase_name, phase_label in [
        ("opening", "1단계: 입론"),
        ("chained_rebuttal", "2단계: 연쇄논박"),
        ("free_rebuttal", "3단계: 자유논박"),
        ("role_reversal", "4단계: 역할반전"),
        ("synthesis", "5단계: 종합 및 재개념화"),
    ]:
        entries = [e for e in state["debate_history"] if e["phase"] == phase_name]
        lines.append(f"{'='*70}")
        lines.append(f" {phase_label} ({len(entries)}건)")
        lines.append(f"{'='*70}")
        for entry in entries:
            s_label = "찬성" if entry["stance"] == "PRO" else "반대"
            speaker = "사용자" if entry["speaker_id"] == "user" else entry["speaker_id"]
            target = entry.get("target_id", "")
            target_str = f" → {target}" if target else ""
            lines.append(f"-" * 70)
            lines.append(f"[턴 {entry['turn']}] {speaker} ({s_label}){target_str}")
            lines.append(entry["content"])
            # 메타정보: tool_calls_log (검색 판단, 쿼리, 약점 분석 등)
            tool_log = entry.get("tool_calls_log", [])
            if tool_log:
                lines.append(f"  [메타] 도구 호출: {tool_log}")
            json_raw = entry.get("json_raw", "")
            if json_raw and speaker != "사용자":
                # 약점 분석이 포함된 경우 표시
                if "약점" in str(entry.get("tool_calls_log", "")):
                    lines.append(f"  [메타] 약점 분석 포함")
                # LLM 원본 응답 길이
                lines.append(f"  [메타] LLM 원본: {len(json_raw)}자")
            lines.append("")

    lines.append(f"{'='*70}")
    lines.append(f" 최적해: {state.get('synthesis_draft', '(없음)')}")
    lines.append(f"{'='*70}")
    lines.append(f"총 발언: {len(state['debate_history'])}건")
    lines.append(f"is_finished: {state.get('is_finished', False)}")
    lines.append(f"{'='*70}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return str(output_path)


def run_single_topic(topic_id: str, user_inputs: dict, config: dict):
    print(f"\n{'#'*70}")
    print(f" 테스트 시작: {topic_id}")
    print(f"{'#'*70}")

    topic_dict = _load_topic(topic_id)
    topic_data = user_inputs["topics"][topic_id]

    personas = create_agents(
        topic=topic_dict,
        debate_format=config["debate_format"],
        user_stance=config["user_stance"],
        agent_intensities=config["agent_intensities"],
    )
    snapshots = [
        AgentSnapshot(
            agent_id=p.agent_id, stance=p.stance, intensity=p.intensity,
            role_description=p.role_description, system_prompt=p.system_prompt,
            focus_area=p.focus_area,
        ) for p in personas
    ]
    state = build_initial_state(
        topic=topic_dict["title"],
        user_stance=config["user_stance"],
        user_intensity=config["user_intensity"],
        agents=snapshots,
        topic_id=topic_dict["id"],
    )

    # ── 1단계: 입론
    print(f"\n[1단계: 입론]")
    state = opening_arguments_node(state)
    state = dict(state)

    opening = topic_data["opening"]
    user_opening = f"### 자기소개와 입장 표명\n{opening['intro']}\n\n### 논거 1\n{opening['arg1']}\n\n### 논거 2\n{opening['arg2']}\n\n### 결론\n{opening['conclusion']}"
    user_turn = len([e for e in state["debate_history"] if e["phase"] == "opening"])
    state["debate_history"].append(DebateEntry(
        turn=user_turn, speaker_id="user", stance=config["user_stance"],
        phase="opening", content=user_opening, target_id=None,
        tool_calls_log=[], json_raw="",
    ))
    state["debate_history"].sort(key=lambda e: e["turn"])
    state["phase"] = "chained_rebuttal"
    print(f"  사용자 입론 삽입 완료")

    # ── 2단계: 연쇄논박
    print(f"\n[2단계: 연쇄논박]")
    state = chained_rebuttal_node(state)
    state = dict(state)

    # 사용자를 공격한 에이전트 찾기
    attacker_id = None
    for e in reversed(state["debate_history"]):
        if e["phase"] == "chained_rebuttal" and e.get("target_id") == "user":
            attacker_id = e["speaker_id"]
            break
    target_id = attacker_id or "agent_1"

    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user", stance=config["user_stance"],
        phase="chained_rebuttal", content=topic_data["rebuttal"], target_id=target_id,
        tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "free_rebuttal"
    print(f"  사용자 연쇄논박 삽입 완료")

    # ── 3단계: 자유논박 (4.5턴 고정)
    print(f"\n[3단계: 자유논박]")
    opposite = "CON" if config["user_stance"] == "PRO" else "PRO"
    opponent = next((a for a in state["agents"] if a["stance"] == opposite), None)
    if opponent:
        state["selected_opponent_id"] = opponent["agent_id"]

    # 턴1: 에이전트 첫 공격
    state = free_rebuttal_node(state)
    state = dict(state)

    for turn_idx, fr_data in enumerate(topic_data["free_rebuttal"]):
        # 사용자 답변+공격
        state["debate_history"].append(DebateEntry(
            turn=state["current_turn"], speaker_id="user", stance=config["user_stance"],
            phase="free_rebuttal", content=fr_data["defense"],
            target_id=state.get("selected_opponent_id"),
            tool_calls_log=[], json_raw="",
        ))
        state["current_turn"] += 1
        state["debate_history"].append(DebateEntry(
            turn=state["current_turn"], speaker_id="user", stance=config["user_stance"],
            phase="free_rebuttal", content=fr_data["attack"],
            target_id=state.get("selected_opponent_id"),
            tool_calls_log=[], json_raw="",
        ))
        state["current_turn"] += 1
        state["free_rebuttal_user_turns"] = turn_idx + 1

        # 에이전트 답변+공격 (마지막 턴은 답변만)
        state = free_rebuttal_node(state)
        state = dict(state)
        print(f"  자유논박 턴 {turn_idx + 1}/2 완료")

    state["phase"] = "role_reversal"

    # ── 4단계: 역할반전
    print(f"\n[4단계: 역할반전]")
    state = role_reversal_node(state)
    state = dict(state)

    reversed_stance = "CON" if config["user_stance"] == "PRO" else "PRO"
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user",
        stance=reversed_stance, phase="role_reversal",
        content=topic_data["role_reversal"], target_id=None,
        tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "synthesis"
    print(f"  사용자 역할반전 삽입 완료")

    # ── 5단계: 종합 회의
    print(f"\n[5단계: 종합 회의]")
    state = synthesis_node(state)
    state = dict(state)

    for syn_idx, syn_text in enumerate(topic_data["synthesis"]):
        state["debate_history"].append(DebateEntry(
            turn=state["current_turn"], speaker_id="user",
            stance=config["user_stance"], phase="synthesis",
            content=syn_text, target_id=None,
            tool_calls_log=[], json_raw="",
        ))
        state["current_turn"] += 1
        state["synthesis_user_turns"] = syn_idx + 1

        state = synthesis_discuss_node(state)
        state = dict(state)
        print(f"  종합 회의 턴 {syn_idx + 1}/2 완료")

    # 최적해 확정
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user",
        stance=config["user_stance"], phase="synthesis",
        content=topic_data["synthesis_final"], target_id=None,
        tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["synthesis_draft"] = topic_data["synthesis_final"]
    state["is_finished"] = True

    # ── 결과 저장
    output_path = _save_results(topic_id, topic_dict["title"], state, topic_data, config)
    print(f"\n  결과 저장: {output_path}")

    return output_path


_FORMAT_CONFIGS = {
    "1:1": {"agent_intensities": [3]},
    "2:2": {"agent_intensities": [3, 2, 4]},
    "3:3": {"agent_intensities": [3, 2, 4, 3, 2]},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default=None, help="특정 토픽만 테스트 (예: tech_001)")
    parser.add_argument("--format", type=str, default="2:2", choices=["1:1", "2:2", "3:3"], help="토론 포맷")
    args = parser.parse_args()

    with _USER_INPUTS_PATH.open(encoding="utf-8") as f:
        user_inputs = json.load(f)

    fmt_config = _FORMAT_CONFIGS[args.format]
    config = {
        "debate_format": args.format,
        "user_stance": user_inputs["user_stance"],
        "user_intensity": user_inputs["user_intensity"],
        "agent_intensities": fmt_config["agent_intensities"],
    }

    if args.topic:
        topics_to_test = [args.topic]
    else:
        topics_to_test = list(user_inputs["topics"].keys())

    results = []
    for topic_id in topics_to_test:
        try:
            path = run_single_topic(topic_id, user_inputs, config)
            results.append((topic_id, "✅ 성공", path))
        except Exception as e:
            results.append((topic_id, f"❌ 실패: {e}", ""))
            import traceback
            traceback.print_exc()

    # ── 요약
    print(f"\n{'='*70}")
    print(f" 전체 테스트 요약 ({len(results)}개 토픽)")
    print(f"{'='*70}")
    for topic_id, status, path in results:
        print(f"  {topic_id}: {status}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
