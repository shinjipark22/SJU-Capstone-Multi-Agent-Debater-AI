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
from typing import Dict, List, Tuple

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
    """'user' 슬롯도 타 AI 에이전트와 **완전히 동일한 조건**으로 수행.

    - system_prompt: persona_factory로 생성 (다른 에이전트와 동일 규칙·도구)
    - 각 스테이지별 발언 생성은 해당 stage 모듈의 _build_*_prompt + _generate_*
      함수를 그대로 호출한다. 별도 축약된 프롬프트를 쓰지 않는다.
    - focus_area는 stage1의 _get_focus_area로 에이전트와 동일 방식으로 할당.
    """

    def __init__(
        self,
        topic: dict,
        stance: str,
        intensity: int = 3,
        topic_id: str = "",
        slot_index: int = 0,
    ):
        """UserProxy 초기화.

        Args:
            slot_index: 같은 진영 내 사용자가 차지할 focus_area 인덱스.
                같은 진영 AI 들이 차지한 인덱스 다음 자리 (남는 자리) 를 받아야
                중복 없이 다양화. 호출자가 같은 진영 AI 수를 계산해 전달한다.
                기본 0 (단독 사용 시 호환).
        """
        from src.phase0.persona_factory import _build_system_prompt
        from src.phase1.stage1_opening.nodes import _get_focus_area

        stance_kr = "찬성" if stance == "PRO" else "반대"
        system_prompt = _build_system_prompt(
            agent_id="user", stance=stance, intensity=intensity,
            title=topic["title"], pro=topic.get("pro", ""),
            con=topic.get("con", ""),
            description=topic.get("description_long"),
        )

        self.topic = topic
        self.topic_id = topic_id
        self.stance = stance
        self.stance_kr = stance_kr
        self.intensity = intensity
        self.focus_area = _get_focus_area(stance, topic_id=topic_id, index=slot_index)

        # 다른 에이전트와 동일한 AgentSnapshot 형태
        self.agent: dict = {
            "agent_id": "user",
            "stance": stance,
            "intensity": intensity,
            "role_description": f"{stance_kr} | 강경도 {intensity}",
            "system_prompt": system_prompt,
            "focus_area": self.focus_area,
        }

    # ── 스테이지별 발언 생성 (타 에이전트와 동일 경로) ────────────────────
    def opening(self, display: str = "찬성1") -> Tuple[str, List[Dict]]:
        """stage1_opening 의 Plan-and-Execute pipeline 사용."""
        from src.phase1.stage1_opening.nodes import _generate_opening
        text, _raw, tc_log = _generate_opening(
            self.agent, self.topic["title"], self.stance, display, self.focus_area,
        )
        return text, tc_log

    def chained_rebuttal(
        self, target_speech: str, target_display: str, history: list,
    ) -> Tuple[str, List[Dict]]:
        """stage2_rebuttal의 동일 경로로 반박 생성."""
        from src.phase1.stage2_rebuttal.nodes import (
            _build_rebuttal_prompt, _generate_rebuttal_speech, _pre_search_rebuttal,
        )
        from src.graph.llm import build_debate_chain
        search_results, _pre_tc = _pre_search_rebuttal(self.topic["title"], target_speech)
        prompt = _build_rebuttal_prompt(
            target_speech, target_display, self.stance_kr,
            search_results=search_results,
        )
        chain = build_debate_chain(history, "user")
        text, _raw, tc_log = _generate_rebuttal_speech(
            self.agent, prompt, target_display, self.stance, chain,
        )
        return text, _pre_tc + tc_log

    def free_defense_attack(
        self, opp_attack: str, opp_opening: str, my_opening: str, history: list,
    ) -> Tuple[str, str, List[Dict]]:
        """stage3_free_rebuttal: 방어 + 공격 한 세트 (타 에이전트 턴과 동일)."""
        from src.phase1.stage3_free_rebuttal.nodes import (
            _build_defense_prompt, _build_attack_prompt, _generate_with_chain,
            _pick_one_argument,
        )
        from src.phase1.stage2_rebuttal.nodes import _pre_search_rebuttal
        from src.graph.llm import build_debate_chain
        from langchain_core.messages import SystemMessage
        # 진영별 URL 중복 제외 컨텍스트 (user stance)
        try:
            from src.graph.vector_store import set_current_stance
            set_current_stance(self.stance)
        except Exception:
            pass

        def _chain():
            msgs = [SystemMessage(content=self.agent["system_prompt"])]
            msgs.extend(build_debate_chain(history, "user"))
            return msgs

        all_tc: List[Dict] = []

        # 자기 (user) 의 직전 자유논박 발언 — 반복 회피 prompt 블록용
        my_previous = next(
            (e.get("content", "") for e in reversed(history)
             if e.get("phase") == "free_rebuttal" and e.get("speaker_id") == "user"),
            "",
        )

        # 방어
        defense_text = ""
        def_tc: List[Dict] = []
        if opp_attack:
            def_sr, def_pre_tc = _pre_search_rebuttal(self.topic["title"], opp_attack)
            all_tc.extend(def_pre_tc)
            dprompt = _build_defense_prompt(
                opp_attack, my_opening, search_results=def_sr, my_previous=my_previous,
            )
            defense_text, _r, def_tc = _generate_with_chain(_chain(), dprompt)
            all_tc.extend(def_tc)

        # 공격
        target = _pick_one_argument(opp_opening) if opp_opening else ""
        atk_sr, atk_pre_tc = _pre_search_rebuttal(self.topic["title"], target) if target else ("", [])
        all_tc.extend(atk_pre_tc)
        aprompt = _build_attack_prompt(
            target, search_results=atk_sr, opp_opening=opp_opening, my_previous=my_previous,
        )
        attack_text, _r2, att_tc = _generate_with_chain(_chain(), aprompt)
        all_tc.extend(att_tc)
        return defense_text, attack_text, all_tc

    def role_reversal(
        self, reversed_stance: str, opponent_openings: str,
    ) -> Tuple[str, List[Dict]]:
        """stage4_role_reversal의 동일 경로."""
        from src.phase1.stage4_role_reversal.nodes import (
            _build_role_reversal_prompt, _generate_role_reversal,
        )
        from src.phase1.stage1_opening.nodes import _pre_search
        # 반전 입장의 focus로 사전 검색
        search_results, pre_tc = _pre_search(self.topic["title"], reversed_stance, self.topic_id)
        prompt = _build_role_reversal_prompt(
            self.topic["title"], reversed_stance, "찬성1" if reversed_stance == "PRO" else "반대1",
            search_results, opponent_openings,
        )
        # 반전된 입장의 agent 스냅샷 (system prompt만 재생성)
        from src.phase0.persona_factory import _build_system_prompt
        rr_agent = {
            **self.agent,
            "stance": reversed_stance,
            "system_prompt": _build_system_prompt(
                agent_id="user", stance=reversed_stance, intensity=self.intensity,
                title=self.topic["title"], pro=self.topic.get("pro", ""),
                con=self.topic.get("con", ""),
                description=self.topic.get("description_long"),
            ),
        }
        text, _raw, tc_log = _generate_role_reversal(rr_agent, prompt)
        return text, pre_tc + tc_log

    def synthesis_proposal(self, history: list) -> Tuple[str, List[Dict]]:
        """stage5_synthesis의 동일 경로."""
        from src.phase1.stage5_synthesis.nodes import _build_proposal_prompt, _build_synthesis_chain
        from src.phase1.stage3_free_rebuttal.nodes import _generate_with_chain
        prompt = _build_proposal_prompt(
            self.topic["title"], self.stance, perspective=self.focus_area, intensity=self.intensity,
        )
        chain = _build_synthesis_chain(self.agent, history, "user")
        text, _raw, tc_log = _generate_with_chain(chain, prompt)
        return text, tc_log

    def synthesis_final(self, history: list) -> Tuple[str, List[Dict]]:
        """마지막 종합 — 합의된 최적해를 명시적으로 확정."""
        from src.phase1.stage5_synthesis.nodes import _build_synthesis_chain
        from src.phase1.stage3_free_rebuttal.nodes import _generate_with_chain
        prompt = (
            "지금까지의 토론과 회의를 종합하여 **양측이 합의할 수 있는 최적해**를 2~3문장으로 "
            "확정하여 선언하세요. 새 쟁점을 추가하지 말고, 앞서 논의된 타협안·조건을 종합해 "
            "구체적 결론 문장으로 표현하세요. "
            "형식: '우리의 최적해는 ~입니다. 이는 ~조건과 ~보완을 전제로 합니다.' 처럼 "
            "명시적으로 '최적해'라는 단어를 포함하세요. 한국어 합니다체."
        )
        chain = _build_synthesis_chain(self.agent, history, "user")
        text, _raw, tc_log = _generate_with_chain(chain, prompt)
        return text, tc_log


def run_experiment(
    topic_id: str,
    debate_format: str,
    output_path: str,
    intensities: str = "",
    preset_key: str = "",
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

    # 강경도: CLI에서 전달받거나 기본값 사용
    if intensities:
        agent_intensities = [int(x) for x in intensities.split(",")]
    else:
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

    # user도 동일 조건의 에이전트로 수행 (시스템 프롬프트·focus_area·도구 동일)
    proxy = UserProxy(topic_dict, user_stance, intensity=3, topic_id=topic_dict["id"])
    # user의 표시명 (찬성N / 반대N) — 같은 진영 에이전트 수 + 1
    _same_stance_count = sum(1 for a in snapshots if a["stance"] == user_stance)
    user_display = f"{'찬성' if user_stance == 'PRO' else '반대'}{_same_stance_count + 1}"
    logger.info("실험 시작: %s / %s (전원 AI, user=%s)", topic_id, debate_format, user_display)

    # ── 1단계: 입론 ──────────────────────────────────────────────────
    state = opening_arguments_node(state)
    state = dict(state)

    user_opening, user_tc = proxy.opening(display=user_display)
    user_turn = len([e for e in state["debate_history"] if e["phase"] == "opening"])
    state["debate_history"].append(DebateEntry(
        turn=user_turn, speaker_id="user", stance=user_stance,
        phase="opening", content=user_opening, target_id=None,
        tool_calls_log=user_tc, json_raw="",
    ))
    state["debate_history"].sort(key=lambda e: e["turn"])
    state["phase"] = "chained_rebuttal"

    # ── 2단계: 연쇄논박 ─────────────────────────────────────────────
    state = chained_rebuttal_node(state)
    state = dict(state)

    attacker_id = None
    ai_attack = ""
    for e in reversed(state["debate_history"]):
        if e["phase"] == "chained_rebuttal" and e.get("target_id") == "user":
            attacker_id = e["speaker_id"]
            ai_attack = e["content"]
            break
    target_id = attacker_id or "agent_1"
    attacker_snap = next((a for a in snapshots if a["agent_id"] == target_id), None)
    target_display = attacker_snap["role_description"].split("|")[0].strip() if attacker_snap else target_id

    user_rebuttal, user_tc = proxy.chained_rebuttal(
        target_speech=ai_attack, target_display=target_display,
        history=state["debate_history"],
    )
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user", stance=user_stance,
        phase="chained_rebuttal", content=user_rebuttal, target_id=target_id,
        tool_calls_log=user_tc, json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "free_rebuttal"

    # ── 3단계: 자유논박 ─────────────────────────────────────────────
    opposite = "CON" if user_stance == "PRO" else "PRO"
    opponent = next((a for a in state["agents"] if a["stance"] == opposite), None)
    if opponent:
        state["selected_opponent_id"] = opponent["agent_id"]

    # user의 자기 입론 (자유 논박 내 방어 근거용)
    my_opening = next(
        (e["content"] for e in state["debate_history"]
         if e["phase"] == "opening" and e["speaker_id"] == "user"),
        "",
    )
    # 상대 입론 (공격 대상)
    opp_opening = next(
        (e["content"] for e in state["debate_history"]
         if e["phase"] == "opening" and e["speaker_id"] == state.get("selected_opponent_id")),
        "",
    )

    # 자유논박 총 6턴: AI 공격(1) + user 방어+공격(2) + AI 방어+공격(2) + user 방어 최종(1)
    state = free_rebuttal_node(state)  # AI 초기 공격 (1턴)
    state = dict(state)

    # 라운드 1: user 방어 + 공격, 이어서 AI 방어 + 공격
    agent_entries = [
        e for e in state["debate_history"]
        if e["speaker_id"] == state.get("selected_opponent_id")
        and e["phase"] == "free_rebuttal"
    ]
    last_agent = agent_entries[-1]["content"] if agent_entries else ""

    user_defense, user_attack, user_tc = proxy.free_defense_attack(
        opp_attack=last_agent, opp_opening=opp_opening,
        my_opening=my_opening, history=state["debate_history"],
    )
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user", stance=user_stance,
        phase="free_rebuttal", content=user_defense,
        target_id=state.get("selected_opponent_id"),
        tool_calls_log=user_tc[: len(user_tc)//2 or len(user_tc)], json_raw="",
    ))
    state["current_turn"] += 1
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user", stance=user_stance,
        phase="free_rebuttal", content=user_attack,
        target_id=state.get("selected_opponent_id"),
        tool_calls_log=user_tc[len(user_tc)//2:], json_raw="",
    ))
    state["current_turn"] += 1
    state["free_rebuttal_user_turns"] = 1

    state = free_rebuttal_node(state)  # AI 방어+공격 (2턴)
    state = dict(state)
    logger.info("[3단계] 자유논박 1라운드 완료")

    # 라운드 2 (최종): user 방어만 (1턴), AI 응답 없음
    agent_entries = [
        e for e in state["debate_history"]
        if e["speaker_id"] == state.get("selected_opponent_id")
        and e["phase"] == "free_rebuttal"
    ]
    last_agent = agent_entries[-1]["content"] if agent_entries else ""
    user_final_defense, _user_atk_discard, user_tc = proxy.free_defense_attack(
        opp_attack=last_agent, opp_opening=opp_opening,
        my_opening=my_opening, history=state["debate_history"],
    )
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user", stance=user_stance,
        phase="free_rebuttal", content=user_final_defense,
        target_id=state.get("selected_opponent_id"),
        tool_calls_log=user_tc[: len(user_tc)//2 or len(user_tc)], json_raw="",
    ))
    state["current_turn"] += 1
    state["free_rebuttal_user_turns"] = 2
    logger.info("[3단계] 자유논박 최종 방어 완료 (총 6턴)")

    state["phase"] = "role_reversal"

    # ── 4단계: 역할반전 ─────────────────────────────────────────────
    state = role_reversal_node(state)
    state = dict(state)

    reversed_stance = "CON" if user_stance == "PRO" else "PRO"
    # 반전된 입장의 기존 입론들
    opp_openings_text = "\n---\n".join(
        e["content"][:300] for e in state["debate_history"]
        if e["phase"] == "opening" and e["stance"] == reversed_stance
    )
    user_rr, user_tc = proxy.role_reversal(
        reversed_stance=reversed_stance, opponent_openings=opp_openings_text,
    )
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user",
        stance=reversed_stance, phase="role_reversal",
        content=user_rr, target_id=None,
        tool_calls_log=user_tc, json_raw="",
    ))
    state["current_turn"] += 1
    state["phase"] = "synthesis"

    # ── 5단계: 종합 ─────────────────────────────────────────────────
    state = synthesis_node(state)
    state = dict(state)

    for syn_idx in range(2):
        user_syn, user_tc = proxy.synthesis_proposal(history=state["debate_history"])
        state["debate_history"].append(DebateEntry(
            turn=state["current_turn"], speaker_id="user",
            stance=user_stance, phase="synthesis",
            content=user_syn, target_id=None,
            tool_calls_log=user_tc, json_raw="",
        ))
        state["current_turn"] += 1
        state["synthesis_user_turns"] = syn_idx + 1

        state = synthesis_discuss_node(state)
        state = dict(state)
        logger.info("[5단계] 종합 턴 %d/2", syn_idx + 1)

    user_final, user_tc = proxy.synthesis_final(history=state["debate_history"])
    state["debate_history"].append(DebateEntry(
        turn=state["current_turn"], speaker_id="user",
        stance=user_stance, phase="synthesis",
        content=user_final, target_id=None,
        tool_calls_log=user_tc, json_raw="",
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
        "intensity_preset": preset_key,
        "agent_intensities": agent_intensities,
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
    parser.add_argument("--intensities", default="", help="강경도 (콤마 구분, 예: 5,1,4)")
    parser.add_argument("--preset", default="", help="프리셋 키 (예: 2v2_polarized)")
    parser.add_argument("--stance", default="PRO", choices=["PRO", "CON"])
    args = parser.parse_args()
    run_experiment(args.topic, args.format, args.output, args.intensities, args.preset, args.stance)


if __name__ == "__main__":
    main()
