"""
_run_single.py -- 단일 실험 실행 (실험 전용 모드)

모든 참여자가 AI. "user" 슬롯도 테스트 대상 모델이 직접 수행한다.
기존 토론 엔진 코드는 수정하지 않고, user 입력만 LLM으로 대체한다.

사용법:
    python -m experiments._run_single \
        --topic tech_001 --format 2:2 \
        --output data/logs/Qwen-2.5-32B-Instruct/tech_001_2v2.json
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
    topics_path = PROJECT_ROOT / "data" / "topics_20260323_processed.json"
    with topics_path.open(encoding="utf-8") as f:
        data = json.load(f)
    for category_topics in data.get("categories", {}).values():
        for t in category_topics:
            if t["id"] == topic_id:
                return t
    raise ValueError(f"topic ID '{topic_id}'를 찾을 수 없습니다.")


class UserProxy:
    """테스트 대상 모델이 user 역할도 수행하는 프록시.

    기존 엔진은 'user' speaker_id를 기대하므로,
    동일 모델로 user 발언을 생성하여 주입한다.
    """

    def __init__(self, topic: dict, stance: str):
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        self._ChatOpenAI = ChatOpenAI
        self._HumanMessage = HumanMessage
        self._SystemMessage = SystemMessage

        self.llm = ChatOpenAI(
            model=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-32B-Instruct-AWQ"),
            base_url=os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"),
            api_key=os.environ.get("LLM_API_KEY", "fake"),
            temperature=0,
            max_tokens=1024,
        )

        stance_kr = "찬성" if stance == "PRO" else "반대"
        my_claim = topic.get("pro", "") if stance == "PRO" else topic.get("con", "")
        opp_claim = topic.get("con", "") if stance == "PRO" else topic.get("pro", "")

        self.system = (
            f"너는 토론 참가자다. {stance_kr} 입장에서 토론한다.\n"
            f"논제: {topic['title']}\n"
            f"너의 주장: {my_claim}\n"
            f"상대 주장(반박 대상): {opp_claim}\n\n"
            f"반드시 한국어 합니다체로 작성하라. 핵심 주장에 **강조** 표시하라.\n"
            f"자료를 인용할 때는 search_web으로 검색한 결과만 사용하라."
        )

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """user 역할의 발언을 생성한다."""
        messages = [
            self._SystemMessage(content=self.system),
            self._HumanMessage(content=prompt),
        ]
        response = self.llm.invoke(messages)
        return response.content if isinstance(response.content, str) else str(response.content)

    def opening(self, ai_openings: str) -> str:
        return self.generate(f"""상대 AI들의 입론:
{ai_openings[:600]}

위를 읽고 입론을 작성하라.

### 자기소개와 입장 표명
(1~2문장)
### 논거 1: 소제목
(3~5줄, 구체적 사례·수치 포함)
### 논거 2: 소제목
(3~5줄, 구체적 사례·수치 포함)
### 결론
(1~2문장)""")

    def rebuttal(self, ai_attack: str) -> str:
        return self.generate(
            f"상대의 공격:\n{ai_attack[:400]}\n\n"
            f"위 공격에서 가장 약한 논거 1개를 골라 3~4문장으로 반박하라.",
            max_tokens=512,
        )

    def free_defense(self, ai_attack: str) -> str:
        return self.generate(
            f"상대의 공격:\n{ai_attack[:300]}\n\n1~2문장으로 방어하라.",
            max_tokens=256,
        )

    def free_attack(self) -> str:
        return self.generate("상대의 논거에서 아직 공격하지 않은 약점을 1~2문장으로 공격하라.", max_tokens=256)

    def role_reversal(self, reversed_stance_kr: str) -> str:
        return self.generate(f"""[역할 반전] 이제 {reversed_stance_kr} 입장에서 주장하라.

### 논거 1: 소제목
(3~4문장)
### 논거 2: 소제목
(3~4문장)
### 결론
(1~2문장)""", max_tokens=512)

    def synthesis(self, ai_opinions: str) -> str:
        return self.generate(
            f"AI 에이전트들의 의견:\n{ai_opinions[:400]}\n\n"
            f"위에 반응하며 최적해를 향한 제안을 1~2문장으로 하라.",
            max_tokens=256,
        )

    def synthesis_final(self) -> str:
        return self.generate(
            "지금까지의 토론을 종합하여 '우리의 최적해'를 2~3문장으로 확정하라.",
            max_tokens=256,
        )


def run_experiment(
    topic_id: str,
    debate_format: str,
    output_path: str,
    user_stance: str = "PRO",
) -> dict:
    """단일 토론 실험. 모든 참여자가 동일 모델(user 포함)."""
    from src.phase0.persona_factory import create_agents
    from src.state import AgentSnapshot, DebateEntry, build_initial_state
    from src.phase1.stage1_opening.nodes import opening_arguments_node
    from src.phase1.stage2_rebuttal.nodes import chained_rebuttal_node
    from src.phase1.stage3_free_rebuttal.nodes import free_rebuttal_node
    from src.phase1.stage4_role_reversal.nodes import role_reversal_node
    from src.phase1.stage5_synthesis.nodes import synthesis_node, synthesis_discuss_node

    start_time = time.time()
    topic_dict = _load_topic(topic_id)

    intensities_map = {"2:2": [3, 2, 4], "3:3": [3, 2, 4, 3, 2]}
    agent_intensities = intensities_map[debate_format]

    personas = create_agents(
        topic=topic_dict, debate_format=debate_format,
        user_stance=user_stance, agent_intensities=agent_intensities,
    )
    snapshots = [
        AgentSnapshot(
            agent_id=p.agent_id, stance=p.stance, intensity=p.intensity,
            role_description=p.role_description, system_prompt=p.system_prompt,
            focus_area=p.focus_area,
        ) for p in personas
    ]
    state = build_initial_state(
        topic=topic_dict["title"], user_stance=user_stance,
        user_intensity=3, agents=snapshots, topic_id=topic_dict["id"],
    )

    # user 프록시 — 동일 모델이 user 역할 수행
    proxy = UserProxy(topic_dict, user_stance)
    logger.info("실험 시작: %s / %s (전원 AI, user_stance=%s)", topic_id, debate_format, user_stance)

    # ── 1단계: 입론 ──────────────────────────────────────────────────
    state = opening_arguments_node(state)
    state = dict(state)

    ai_openings = "\n---\n".join(
        e["content"][:300] for e in state["debate_history"] if e["phase"] == "opening"
    )
    user_opening = proxy.opening(ai_openings)
    user_turn = len([e for e in state["debate_history"] if e["phase"] == "opening"])
    state["debate_history"].append(DebateEntry(
        turn=user_turn, speaker_id="user", stance=user_stance,
        phase="opening", content=user_opening, target_id=None,
        tool_calls_log=[], json_raw="",
    ))
    state["debate_history"].sort(key=lambda e: e["turn"])
    state["phase"] = "chained_rebuttal"

    # ── 2단계: 연쇄논박 ─────────────────────────────────────────────
    state = chained_rebuttal_node(state)
    state = dict(state)

    attacker_id = None
    for e in reversed(state["debate_history"]):
        if e["phase"] == "chained_rebuttal" and e.get("target_id") == "user":
            attacker_id = e["speaker_id"]
            break
    target_id = attacker_id or "agent_1"

    ai_attack = ""
    for e in reversed(state["debate_history"]):
        if e["phase"] == "chained_rebuttal" and e.get("target_id") == "user":
            ai_attack = e["content"]
            break

    user_rebuttal = proxy.rebuttal(ai_attack)
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user", stance=user_stance,
        phase="chained_rebuttal", content=user_rebuttal, target_id=target_id,
        tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "free_rebuttal"

    # ── 3단계: 자유논박 ─────────────────────────────────────────────
    opposite = "CON" if user_stance == "PRO" else "PRO"
    opponent = next((a for a in state["agents"] if a["stance"] == opposite), None)
    if opponent:
        state["selected_opponent_id"] = opponent["agent_id"]

    state = free_rebuttal_node(state)
    state = dict(state)

    for turn_idx in range(2):
        agent_entries = [
            e for e in state["debate_history"]
            if e["speaker_id"] == state.get("selected_opponent_id")
            and e["phase"] == "free_rebuttal"
        ]
        last_agent = agent_entries[-1]["content"] if agent_entries else ""

        user_defense = proxy.free_defense(last_agent)
        state["debate_history"].append(DebateEntry(
            turn=state["current_turn"], speaker_id="user", stance=user_stance,
            phase="free_rebuttal", content=user_defense,
            target_id=state.get("selected_opponent_id"),
            tool_calls_log=[], json_raw="",
        ))
        state["current_turn"] += 1

        user_attack = proxy.free_attack()
        state["debate_history"].append(DebateEntry(
            turn=state["current_turn"], speaker_id="user", stance=user_stance,
            phase="free_rebuttal", content=user_attack,
            target_id=state.get("selected_opponent_id"),
            tool_calls_log=[], json_raw="",
        ))
        state["current_turn"] += 1
        state["free_rebuttal_user_turns"] = turn_idx + 1

        state = free_rebuttal_node(state)
        state = dict(state)
        logger.info("[3단계] 자유논박 턴 %d/2", turn_idx + 1)

    state["phase"] = "role_reversal"

    # ── 4단계: 역할반전 ─────────────────────────────────────────────
    state = role_reversal_node(state)
    state = dict(state)

    reversed_stance = "CON" if user_stance == "PRO" else "PRO"
    reversed_kr = "반대" if user_stance == "PRO" else "찬성"
    user_rr = proxy.role_reversal(reversed_kr)
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user",
        stance=reversed_stance, phase="role_reversal",
        content=user_rr, target_id=None,
        tool_calls_log=[], json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "synthesis"

    # ── 5단계: 종합 ─────────────────────────────────────────────────
    state = synthesis_node(state)
    state = dict(state)

    for syn_idx in range(2):
        ai_opinions = "\n".join(
            f"[{e['speaker_id']}] {e['content'][:100]}"
            for e in state["debate_history"]
            if e["phase"] == "synthesis" and e["speaker_id"] != "user"
        )[-500:]

        user_syn = proxy.synthesis(ai_opinions)
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
        logger.info("[5단계] 종합 턴 %d/2", syn_idx + 1)

    user_final = proxy.synthesis_final()
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user",
        stance=user_stance, phase="synthesis",
        content=user_final, target_id=None,
        tool_calls_log=[], json_raw="",
    ))
    state["synthesis_draft"] = user_final
    state["is_finished"] = True

    elapsed = time.time() - start_time

    # ── JSON 출력 ────────────────────────────────────────────────────
    result = {
        "topic_id": topic_id,
        "topic": topic_dict["title"],
        "category": topic_dict.get("category", ""),
        "user_stance": user_stance,
        "debate_format": debate_format,
        "model_name": os.environ.get("LLM_MODEL", "unknown"),
        "experiment_mode": "all_ai",
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--format", required=True, choices=["2:2", "3:3"])
    parser.add_argument("--output", required=True)
    parser.add_argument("--stance", default="PRO", choices=["PRO", "CON"])
    args = parser.parse_args()
    run_experiment(args.topic, args.format, args.output, args.stance)


if __name__ == "__main__":
    main()
