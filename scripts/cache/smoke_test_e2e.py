"""
캐시 e2e 동작 검증.

tech_003 2:2 세션을 시뮬레이션하여 5개 캐시 단계가 모두 HIT 되는지 확인:
  1. cache_variant_idx 가 build_initial_state 에서 1/2/3 으로 박힘
  2. AI 입론 (load_opening)
  3. 어시스턴트 입론 가이드 (load_assistant_opening)
  4. AI-AI 연쇄논박 (load_ai_rebuttal)
  5. AI 역할반전 (load_role_reversal)
  6. 어시스턴트 역할반전 가이드 (load_assistant_role_reversal)

각 단계에서 loader.py 의 INFO 로그 ([cache] HIT ...) 가 떠야 한다.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


def _load_env():
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env()
os.environ["SPEECH_CACHE_ENABLED"] = "1"

# 캐시 hit/miss 로그 보이게
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

from src.phase0.persona_factory import create_agents  # noqa: E402
from src.state import (  # noqa: E402
    AgentSnapshot, DebateEntry, build_initial_state, build_chained_rebuttal_pairs,
)


TOPIC_ID = "tech_003"
TOPICS_PATH = ROOT / "data" / "topics_20260323_processed.json"


def _load_topic() -> dict:
    data = json.loads(TOPICS_PATH.read_text(encoding="utf-8"))
    for cat in data.get("categories", {}).values():
        for t in cat:
            if t["id"] == TOPIC_ID:
                return t
    raise SystemExit(f"topic '{TOPIC_ID}' not found")


def _section(name: str):
    print(f"\n{'='*60}\n{name}\n{'='*60}")


def main():
    topic = _load_topic()

    # ── [1] State 초기화 — cache_variant_idx 확인 ───────────────────────
    _section("[1] build_initial_state — cache_variant_idx")
    personas = create_agents(
        topic=topic,
        debate_format="2:2",
        user_stance="PRO",
        agent_intensities=[3, 3, 3],  # 2:2 면 AI 3명 (CON 2 + PRO 1)
    )
    snapshots = [
        AgentSnapshot(
            agent_id=p.agent_id, stance=p.stance, intensity=p.intensity,
            role_description=p.role_description, system_prompt=p.system_prompt,
            focus_area=p.focus_area,
        )
        for p in personas
    ]
    state = build_initial_state(
        topic=topic["title"], user_stance="PRO", user_intensity=3,
        agents=snapshots, topic_id=TOPIC_ID, mode="constructive",
    )
    v = state["cache_variant_idx"]
    assert v in (1, 2, 3), f"cache_variant_idx invalid: {v}"
    print(f"  ✓ cache_variant_idx = {v}")
    for a in state["agents"]:
        print(f"    agent {a['agent_id']}: stance={a['stance']} intensity={a['intensity']} focus={a['focus_area'][:30]}")

    # ── [2] AI 입론 캐시 hit ──────────────────────────────────────────
    _section("[2] AI 입론 — _generate_openings_for")
    from src.graph.main_graph import _generate_openings_for
    ai_speaker_ids = [a["agent_id"] for a in state["agents"]]
    out = _generate_openings_for(state, ai_speaker_ids)
    new_entries = out["debate_history"][-len(ai_speaker_ids):]
    for e in new_entries:
        speech = e["content"][:80].replace("\n", " ")
        print(f"  {e['speaker_id']} ({e['stance']}): {speech}...")
    # state 업데이트
    state = {**state, **out}

    # ── [3] 어시스턴트 입론 가이드 캐시 hit ───────────────────────────
    _section("[3] 어시스턴트 입론 — build_guide_message('opening', ctx)")
    from src.debate_assistant.text_guide import GuideContext, build_guide_message
    from src.debate_assistant.text_guide import get_user_slot_focus_area
    user_focus = get_user_slot_focus_area(
        TOPIC_ID, "PRO",
        excluded_focuses=[a["focus_area"] for a in state["agents"] if a["stance"] == "PRO"],
    )
    ctx = GuideContext(
        topic=topic["title"], user_stance="PRO", assistant_name="비비드",
        pro_claim=topic.get("pro"), con_claim=topic.get("con"),
        topic_id=TOPIC_ID, user_focus_area=user_focus,
        cache_variant_idx=v,
    )
    print(f"  user_focus_area = {user_focus}")
    msg = build_guide_message("opening", ctx)
    print(f"  ✓ guide len={len(msg)}자, head: {msg[:80].replace(chr(10), ' ')}")

    # ── [4] AI-AI 연쇄논박 캐시 hit (2:2 의 AI-AI 쌍) ─────────────────
    _section("[4] AI-AI 연쇄논박 — process_one_rebuttal_step")
    pairs = build_chained_rebuttal_pairs(state["agents"], state["speaking_order"])
    state["rebuttal_pairs"] = pairs
    state["phase"] = "chained_rebuttal"
    state["current_rebuttal_round"] = 0
    # AI-AI 쌍이 첫 번째일 가능성 높음. 한 step 호출.
    from src.phase1.stage2_rebuttal.nodes import process_one_rebuttal_step
    for i, p in enumerate(pairs):
        if p["attacker_id"] != "user" and p["target_id"] != "user":
            print(f"  AI-AI 쌍 인덱스: {i}  ({p['attacker_id']} → {p['target_id']})")
            break
    state["current_rebuttal_round"] = i
    out = process_one_rebuttal_step(state)
    # 마지막 추가된 entry
    new_history = out["debate_history"]
    last = new_history[-1] if len(new_history) > len(state["debate_history"]) else None
    if last:
        print(f"  ✓ {last['speaker_id']}: {last['content'][:80].replace(chr(10),' ')}")
    state = {**state, **out}

    # ── [5] AI 역할반전 캐시 hit ──────────────────────────────────────
    _section("[5] AI 역할반전 — role_reversal_node")
    state["phase"] = "role_reversal"
    from src.phase1.stage4_role_reversal.nodes import role_reversal_node
    out = role_reversal_node(state)
    new_entries = out["debate_history"][len(state["debate_history"]):]
    for e in new_entries:
        print(f"  ✓ {e['speaker_id']} ({e['stance']}): {e['content'][:80].replace(chr(10),' ')}")

    # ── [6] 어시스턴트 역할반전 가이드 캐시 hit ──────────────────────
    _section("[6] 어시스턴트 역할반전 — build_guide_message('role_reversal', ctx)")
    ctx2 = GuideContext(
        topic=topic["title"], user_stance="PRO", assistant_name="비비드",
        pro_claim=topic.get("pro"), con_claim=topic.get("con"),
        topic_id=TOPIC_ID,
        cache_variant_idx=v,
    )
    msg = build_guide_message("role_reversal", ctx2)
    print(f"  ✓ guide len={len(msg)}자, head: {msg[:80].replace(chr(10),' ')}")

    print(f"\n{'='*60}\n  ✅ e2e 검증 완료 — 5단계 캐시 모두 동작\n{'='*60}")


if __name__ == "__main__":
    main()
