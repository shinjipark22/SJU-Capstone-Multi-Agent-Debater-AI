"""
_run_single.py -- 단일 실험 실행 (서브프로세스에서 호출)

환경변수로 모델/엔드포인트가 설정된 상태에서 실행되며,
src.* 모듈을 임포트하여 토론을 수행하고 JSON으로 저장한다.

사용법:
    python -m experiments._run_single \
        --topic tech_001 --stance PRO --format 2:2 \
        --output data/logs/Qwen-2.5-32B-Instruct/tech_001_PRO_2v2.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 프로젝트 루트를 path에 추가 (서브프로세스에서 실행 시 필요)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_topic(topic_id: str) -> dict:
    """topics JSON에서 단일 토픽을 로드한다."""
    topics_path = PROJECT_ROOT / "data" / "topics_20260323_processed.json"
    with topics_path.open(encoding="utf-8") as f:
        data = json.load(f)
    for category_topics in data.get("categories", {}).values():
        for t in category_topics:
            if t["id"] == topic_id:
                return t
    raise ValueError(f"topic ID '{topic_id}'를 찾을 수 없습니다.")


def _load_user_inputs() -> dict:
    """사용자 입력 데이터를 로드한다."""
    user_inputs_path = PROJECT_ROOT / "tests" / "user_inputs_all_topics.json"
    with user_inputs_path.open(encoding="utf-8") as f:
        return json.load(f)


def run_experiment(
    topic_id: str,
    user_stance: str,
    debate_format: str,
    output_path: str,
) -> dict:
    """단일 토론 실험을 실행하고 결과를 JSON으로 저장한다."""
    from src.phase0.persona_factory import create_agents
    from src.state import AgentSnapshot, DebateEntry, build_initial_state
    from src.phase1.stage1_opening.nodes import opening_arguments_node
    from src.phase1.stage2_rebuttal.nodes import chained_rebuttal_node
    from src.phase1.stage3_free_rebuttal.nodes import free_rebuttal_node
    from src.phase1.stage4_role_reversal.nodes import role_reversal_node
    from src.phase1.stage5_synthesis.nodes import synthesis_node, synthesis_discuss_node

    start_time = time.time()

    topic_dict = _load_topic(topic_id)
    user_inputs = _load_user_inputs()
    topic_data = user_inputs["topics"][topic_id]

    # user_inputs는 PRO 기준으로 작성됨.
    # CON 실험 시에도 동일 mock 데이터를 사용 (사용자 입력은 평가 대상 아님)
    # 실제 서비스에서는 사용자가 직접 입력하므로 mock 데이터 stance 불일치는 무관

    # 강경도 설정
    intensities_map = {"2:2": [3, 2, 4], "3:3": [3, 2, 4, 3, 2]}
    agent_intensities = intensities_map[debate_format]

    personas = create_agents(
        topic=topic_dict,
        debate_format=debate_format,
        user_stance=user_stance,
        agent_intensities=agent_intensities,
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
        user_stance=user_stance,
        user_intensity=user_inputs.get("user_intensity", 3),
        agents=snapshots,
        topic_id=topic_dict["id"],
    )

    # ── 1단계: 입론
    state = opening_arguments_node(state)
    state = dict(state)

    opening = topic_data["opening"]
    user_opening = (
        f"### 자기소개와 입장 표명\n{opening['intro']}\n\n"
        f"### 논거 1\n{opening['arg1']}\n\n"
        f"### 논거 2\n{opening['arg2']}\n\n"
        f"### 결론\n{opening['conclusion']}"
    )
    user_turn = len([e for e in state["debate_history"] if e["phase"] == "opening"])
    state["debate_history"].append(DebateEntry(
        turn=user_turn, speaker_id="user", stance=user_stance,
        phase="opening", content=user_opening, target_id=None,
        tool_calls_log=[], json_raw="",
    ))
    state["debate_history"].sort(key=lambda e: e["turn"])
    state["phase"] = "chained_rebuttal"

    # ── 2단계: 연쇄논박
    state = chained_rebuttal_node(state)
    state = dict(state)

    attacker_id = None
    for e in reversed(state["debate_history"]):
        if e["phase"] == "chained_rebuttal" and e.get("target_id") == "user":
            attacker_id = e["speaker_id"]
            break
    target_id = attacker_id or "agent_1"

    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user", stance=user_stance,
        phase="chained_rebuttal", content=topic_data["rebuttal"], target_id=target_id,
        tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "free_rebuttal"

    # ── 3단계: 자유논박 (4.5턴)
    opposite = "CON" if user_stance == "PRO" else "PRO"
    opponent = next((a for a in state["agents"] if a["stance"] == opposite), None)
    if opponent:
        state["selected_opponent_id"] = opponent["agent_id"]

    state = free_rebuttal_node(state)
    state = dict(state)

    for turn_idx, fr_data in enumerate(topic_data["free_rebuttal"]):
        state["debate_history"].append(DebateEntry(
            turn=state["current_turn"], speaker_id="user", stance=user_stance,
            phase="free_rebuttal", content=fr_data["defense"],
            target_id=state.get("selected_opponent_id"),
            tool_calls_log=[], json_raw="",
        ))
        state["current_turn"] += 1
        state["debate_history"].append(DebateEntry(
            turn=state["current_turn"], speaker_id="user", stance=user_stance,
            phase="free_rebuttal", content=fr_data["attack"],
            target_id=state.get("selected_opponent_id"),
            tool_calls_log=[], json_raw="",
        ))
        state["current_turn"] += 1
        state["free_rebuttal_user_turns"] = turn_idx + 1

        state = free_rebuttal_node(state)
        state = dict(state)

    state["phase"] = "role_reversal"

    # ── 4단계: 역할반전
    state = role_reversal_node(state)
    state = dict(state)

    reversed_stance = "CON" if user_stance == "PRO" else "PRO"
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user",
        stance=reversed_stance, phase="role_reversal",
        content=topic_data["role_reversal"], target_id=None,
        tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "synthesis"

    # ── 5단계: 종합 회의
    state = synthesis_node(state)
    state = dict(state)

    for syn_idx, syn_text in enumerate(topic_data["synthesis"]):
        state["debate_history"].append(DebateEntry(
            turn=state["current_turn"], speaker_id="user",
            stance=user_stance, phase="synthesis",
            content=syn_text, target_id=None,
            tool_calls_log=[], json_raw="",
        ))
        state["current_turn"] += 1
        state["synthesis_user_turns"] = syn_idx + 1

        state = synthesis_discuss_node(state)
        state = dict(state)

    # 최적해 확정
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user",
        stance=user_stance, phase="synthesis",
        content=topic_data["synthesis_final"], target_id=None,
        tool_calls_log=[], json_raw="",
    ))
    state["synthesis_draft"] = topic_data["synthesis_final"]
    state["is_finished"] = True

    elapsed = time.time() - start_time

    # ── JSON 출력 구성
    result = {
        "topic_id": topic_id,
        "topic": topic_dict["title"],
        "category": topic_dict.get("category", ""),
        "user_stance": user_stance,
        "debate_format": debate_format,
        "model_name": os.environ.get("LLM_MODEL", "unknown"),
        "created_at": datetime.now().isoformat(),
        "duration_seconds": round(elapsed, 1),
        "is_finished": state.get("is_finished", False),
        "turns": [],
    }

    for entry in state["debate_history"]:
        turn_data = {
            "turn_id": entry["turn"],
            "speaker": entry["speaker_id"],
            "side": entry["stance"],
            "phase": entry["phase"],
            "text": entry["content"],
            "target_id": entry.get("target_id"),
            "tool_calls": [],
        }
        for tc in entry.get("tool_calls_log", []):
            if tc.get("name") == "search_web":
                turn_data["tool_calls"].append({
                    "name": "search_web",
                    "query": tc.get("args", {}).get("query", ""),
                    "results": [tc.get("result", "")],
                })
            elif tc.get("name") == "analyze_weakness":
                turn_data["tool_calls"].append({
                    "name": "analyze_weakness",
                    "query": "",
                    "results": [tc.get("result", "")],
                })
        result["turns"].append(turn_data)

    # 저장
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("실험 완료: %s (%.1f초) → %s", topic_id, elapsed, output_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="단일 토론 실험 실행")
    parser.add_argument("--topic", required=True, help="토픽 ID (예: tech_001)")
    parser.add_argument("--stance", required=True, choices=["PRO", "CON"])
    parser.add_argument("--format", required=True, choices=["2:2", "3:3"])
    parser.add_argument("--output", required=True, help="JSON 출력 경로")
    args = parser.parse_args()

    run_experiment(
        topic_id=args.topic,
        user_stance=args.stance,
        debate_format=args.format,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
