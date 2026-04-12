"""
_run_single.py -- 단일 실험 실행 (서브프로세스에서 호출)

환경변수로 모델/엔드포인트가 설정된 상태에서 실행되며,
GPT-4o-mini가 사용자 역할을 동적으로 수행한다 (Dynamic Multi-agent Environment).

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


def run_experiment(
    topic_id: str,
    user_stance: str,
    debate_format: str,
    output_path: str,
) -> dict:
    """단일 토론 실험을 실행한다. 사용자 역할은 GPT-4o-mini가 동적 수행."""
    from experiments.user_agent import UserAgent
    from src.phase0.persona_factory import create_agents
    from src.state import AgentSnapshot, DebateEntry, build_initial_state
    from src.phase1.stage1_opening.nodes import opening_arguments_node
    from src.phase1.stage2_rebuttal.nodes import chained_rebuttal_node
    from src.phase1.stage3_free_rebuttal.nodes import free_rebuttal_node
    from src.phase1.stage4_role_reversal.nodes import role_reversal_node
    from src.phase1.stage5_synthesis.nodes import synthesis_node, synthesis_discuss_node

    start_time = time.time()
    topic_dict = _load_topic(topic_id)

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
        user_intensity=3,
        agents=snapshots,
        topic_id=topic_dict["id"],
    )

    # ── 사용자 대행 에이전트 초기화 (GPT-4o-mini, temp=0, seed=42)
    user = UserAgent(
        topic_id=topic_id,
        topic_title=topic_dict["title"],
        stance=user_stance,
        pro_claim=topic_dict.get("pro", ""),
        con_claim=topic_dict.get("con", ""),
    )
    logger.info("UserAgent 초기화: %s / %s / %s", topic_id, user_stance, debate_format)

    # ══════════════════════════════════════════════════════════════════
    # 1단계: 입론
    # ══════════════════════════════════════════════════════════════════
    state = opening_arguments_node(state)
    state = dict(state)

    # AI 입론들을 사용자 대행에게 전달 (맥락)
    for e in state["debate_history"]:
        if e["phase"] == "opening" and e["speaker_id"] != "user":
            user.add_ai_context(e["content"][:300], e["speaker_id"])

    # 사용자 입론 — GPT-4o-mini 동적 생성
    user_opening = user.generate_opening()
    user_turn = len([e for e in state["debate_history"] if e["phase"] == "opening"])
    state["debate_history"].append(DebateEntry(
        turn=user_turn, speaker_id="user", stance=user_stance,
        phase="opening", content=user_opening, target_id=None,
        tool_calls_log=[], json_raw="",
    ))
    state["debate_history"].sort(key=lambda e: e["turn"])
    state["phase"] = "chained_rebuttal"
    logger.info("[1단계] 사용자 입론 생성 완료")

    # ══════════════════════════════════════════════════════════════════
    # 2단계: 연쇄논박
    # ══════════════════════════════════════════════════════════════════
    state = chained_rebuttal_node(state)
    state = dict(state)

    # 사용자를 공격한 에이전트 찾기
    attacker_id = None
    for e in reversed(state["debate_history"]):
        if e["phase"] == "chained_rebuttal" and e.get("target_id") == "user":
            attacker_id = e["speaker_id"]
            break
    target_id = attacker_id or "agent_1"

    # 사용자 연쇄논박 — AI의 공격에 동적 반응
    ai_attack_on_user = ""
    for e in reversed(state["debate_history"]):
        if e["phase"] == "chained_rebuttal" and e.get("target_id") == "user":
            ai_attack_on_user = e["content"]
            break

    user_rebuttal = user.generate_rebuttal(ai_attack_on_user)
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user", stance=user_stance,
        phase="chained_rebuttal", content=user_rebuttal, target_id=target_id,
        tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "free_rebuttal"
    logger.info("[2단계] 사용자 연쇄논박 생성 완료")

    # ══════════════════════════════════════════════════════════════════
    # 3단계: 자유논박 (4.5턴)
    # ══════════════════════════════════════════════════════════════════
    opposite = "CON" if user_stance == "PRO" else "PRO"
    opponent = next((a for a in state["agents"] if a["stance"] == opposite), None)
    if opponent:
        state["selected_opponent_id"] = opponent["agent_id"]

    # 턴1: 에이전트 첫 공격
    state = free_rebuttal_node(state)
    state = dict(state)

    for turn_idx in range(2):
        # 에이전트의 마지막 공격 추출
        agent_entries = [
            e for e in state["debate_history"]
            if e["speaker_id"] == state.get("selected_opponent_id")
            and e["phase"] == "free_rebuttal"
        ]
        last_agent_attack = agent_entries[-1]["content"] if agent_entries else ""

        # 사용자 방어 — AI 공격에 동적 반응
        user_defense = user.generate_free_rebuttal_defense(last_agent_attack)
        state["debate_history"].append(DebateEntry(
            turn=state["current_turn"], speaker_id="user", stance=user_stance,
            phase="free_rebuttal", content=user_defense,
            target_id=state.get("selected_opponent_id"),
            tool_calls_log=[], json_raw="",
        ))
        state["current_turn"] += 1

        # 사용자 공격 — 동적 생성
        user_attack = user.generate_free_rebuttal_attack()
        state["debate_history"].append(DebateEntry(
            turn=state["current_turn"], speaker_id="user", stance=user_stance,
            phase="free_rebuttal", content=user_attack,
            target_id=state.get("selected_opponent_id"),
            tool_calls_log=[], json_raw="",
        ))
        state["current_turn"] += 1
        state["free_rebuttal_user_turns"] = turn_idx + 1

        # 에이전트 답변+공격 (마지막 턴은 답변만)
        state = free_rebuttal_node(state)
        state = dict(state)
        logger.info("[3단계] 자유논박 턴 %d/2 완료", turn_idx + 1)

    state["phase"] = "role_reversal"

    # ══════════════════════════════════════════════════════════════════
    # 4단계: 역할반전
    # ══════════════════════════════════════════════════════════════════
    state = role_reversal_node(state)
    state = dict(state)

    # 사용자 역할반전 — 동적 생성
    reversed_stance = "CON" if user_stance == "PRO" else "PRO"
    user_role_reversal = user.generate_role_reversal()
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user",
        stance=reversed_stance, phase="role_reversal",
        content=user_role_reversal, target_id=None,
        tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "synthesis"
    logger.info("[4단계] 사용자 역할반전 생성 완료")

    # ══════════════════════════════════════════════════════════════════
    # 5단계: 종합 회의
    # ══════════════════════════════════════════════════════════════════
    state = synthesis_node(state)
    state = dict(state)

    for syn_idx in range(2):
        # AI 에이전트들의 최근 발언 수집
        ai_opinions = "\n".join(
            f"[{e['speaker_id']}] {e['content'][:100]}"
            for e in state["debate_history"]
            if e["phase"] == "synthesis" and e["speaker_id"] != "user"
        )[-500:]

        # 사용자 종합 발언 — 동적 생성
        user_syn = user.generate_synthesis(ai_opinions)
        state["debate_history"].append(DebateEntry(
            turn=state["current_turn"], speaker_id="user",
            stance=user_stance, phase="synthesis",
            content=user_syn, target_id=None,
            tool_calls_log=[], json_raw="",
        ))
        state["current_turn"] += 1
        state["synthesis_user_turns"] = syn_idx + 1

        state = synthesis_discuss_node(state)
        state = dict(state)
        logger.info("[5단계] 종합 회의 턴 %d/2 완료", syn_idx + 1)

    # 최적해 확정
    user_final = user.generate_synthesis_final()
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user",
        stance=user_stance, phase="synthesis",
        content=user_final, target_id=None,
        tool_calls_log=[], json_raw="",
    ))
    state["synthesis_draft"] = user_final
    state["is_finished"] = True

    elapsed = time.time() - start_time

    # ══════════════════════════════════════════════════════════════════
    # JSON 출력
    # ══════════════════════════════════════════════════════════════════
    result = {
        "topic_id": topic_id,
        "topic": topic_dict["title"],
        "category": topic_dict.get("category", ""),
        "user_stance": user_stance,
        "debate_format": debate_format,
        "model_name": os.environ.get("LLM_MODEL", "unknown"),
        "user_agent": "gpt-4o-mini (temp=0, seed=42)",
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

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info("실험 완료: %s (%.1f초) → %s", topic_id, elapsed, output_path)
    return result


def main():
    parser = argparse.ArgumentParser(description="단일 토론 실험 실행")
    parser.add_argument("--topic", required=True)
    parser.add_argument("--stance", required=True, choices=["PRO", "CON"])
    parser.add_argument("--format", required=True, choices=["2:2", "3:3"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    run_experiment(args.topic, args.stance, args.format, args.output)


if __name__ == "__main__":
    main()
