"""
AI-AI 연쇄논박 캐시 사전 생성 스크립트.

variant N 의 연쇄논박은 variant N 의 입론 캐시를 history 로 주입해 생성한다
(variant correlation). 따라서 generate_opening_cache 가 먼저 실행돼 있어야 한다.

저장:
  data/cache/ai_rebuttals/{topic}_{attacker_stance}_af{att_focus}_tf{tgt_focus}__v{N}.json

사용:
  python scripts/cache/generate_ai_rebuttal_cache.py --topic tech_003
  python scripts/cache/generate_ai_rebuttal_cache.py --topic tech_003 --variant-idx 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))


def _load_env() -> None:
    env_path = _ROOT / ".env"
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
os.environ["SPEECH_CACHE_ENABLED"] = "0"


from src.phase0.persona_factory import _build_system_prompt  # noqa: E402
from src.phase1.stage2_rebuttal.nodes import generate_ai_rebuttal  # noqa: E402
from src.state import DebateEntry  # noqa: E402


_TOPICS_PATH = _ROOT / "data" / "topics_20260323_processed.json"
_OPENING_CACHE_DIR = _ROOT / "data" / "cache" / "openings"
_RBUT_CACHE_DIR = _ROOT / "data" / "cache" / "ai_rebuttals"

CACHE_PROMPT_VERSION = "v1"
DEFAULT_INTENSITY = 3
ALL_VARIANTS = (1, 2, 3)


def _load_topic(topic_id: str) -> dict:
    with _TOPICS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    for cat in data.get("categories", {}).values():
        for t in cat:
            if t["id"] == topic_id:
                return t
    raise SystemExit(f"topic_id '{topic_id}' not found")


def _load_opening_speech(
    topic_id: str, stance: str, intensity: int, focus_idx: int, variant_idx: int,
) -> str:
    """variant_idx 매칭되는 입론 캐시에서 speech 추출."""
    path = _OPENING_CACHE_DIR / f"{topic_id}_{stance}_i{intensity}_f{focus_idx}__v{variant_idx}.json"
    if not path.exists():
        raise SystemExit(
            f"입론 캐시 없음: {path}\n"
            f"먼저 generate_opening_cache 실행 필요 (variant_idx={variant_idx})."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    speech = data.get("speech")
    if not speech:
        raise SystemExit(f"입론 캐시 speech 누락: {path}")
    return speech


def _build_agent(topic: dict, stance: str, intensity: int) -> dict:
    return {
        "agent_id": "agent_cache_att",
        "stance": stance,
        "intensity": intensity,
        "system_prompt": _build_system_prompt(
            agent_id="agent_cache_att",
            stance=stance,
            intensity=intensity,
            title=topic["title"],
            pro=topic["pro"],
            con=topic["con"],
            description=topic.get("description_long"),
        ),
        "focus_area": "",  # 반박 단계엔 미사용
    }


def _cache_path(
    topic_id: str, attacker_stance: str, att_focus: int, tgt_focus: int, variant_idx: int,
) -> Path:
    return _RBUT_CACHE_DIR / (
        f"{topic_id}_{attacker_stance}_af{att_focus}_tf{tgt_focus}__v{variant_idx}.json"
    )


def _generate_one(
    topic: dict, attacker_stance: str, att_focus: int, tgt_focus: int, variant_idx: int,
) -> dict:
    target_stance = "CON" if attacker_stance == "PRO" else "PRO"

    # variant_idx 매칭되는 입론을 history 로
    attacker_opening = _load_opening_speech(
        topic["id"], attacker_stance, DEFAULT_INTENSITY, att_focus, variant_idx,
    )
    target_opening = _load_opening_speech(
        topic["id"], target_stance, DEFAULT_INTENSITY, tgt_focus, variant_idx,
    )

    history = [
        DebateEntry(
            turn=0, speaker_id="agent_cache_tgt", stance=target_stance,
            phase="opening", content=target_opening, target_id=None,
        ),
        DebateEntry(
            turn=1, speaker_id="agent_cache_att", stance=attacker_stance,
            phase="opening", content=attacker_opening, target_id=None,
        ),
    ]

    attacker_agent = _build_agent(topic, attacker_stance, DEFAULT_INTENSITY)

    t0 = time.time()
    entry = generate_ai_rebuttal(
        topic=topic["title"], history=history, agent=attacker_agent,
        target_id="agent_cache_tgt",
        stance_num=1, target_stance_num=1,
        current_turn=2, is_response=False,
    )
    dt = time.time() - t0
    speech = entry["content"]
    tool_calls = entry.get("tool_calls_log", []) or []
    excerpts = []
    for tc in tool_calls:
        if tc.get("name") != "search_web":
            continue
        excerpts.append({
            "query": (tc.get("args") or {}).get("query", ""),
            "content": str(tc.get("result", ""))[:2000],
        })
    print(f"  완료 ({dt:.1f}s, {len(speech)}자, search={len(excerpts)})", flush=True)

    return {
        "prompt_version": CACHE_PROMPT_VERSION,
        "topic_id": topic["id"],
        "attacker_stance": attacker_stance,
        "attacker_focus_idx": att_focus,
        "target_focus_idx": tgt_focus,
        "intensity": DEFAULT_INTENSITY,
        "variant_idx": variant_idx,
        "speech": speech,
        "search_excerpts": excerpts,
        "generated_at": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--variant-idx", type=int, choices=[1, 2, 3])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    topic = _load_topic(args.topic)
    print(f"\n[AI-AI 연쇄논박 캐시] 토픽: {topic['id']} — {topic['title']}\n")

    _RBUT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    variants = [args.variant_idx] if args.variant_idx else list(ALL_VARIANTS)

    # 2:2 케이스의 AI-AI 쌍 — focus_idx 0 끼리 (PRO_f0↔CON_f0)
    base_cases = [
        ("PRO", 0, 0),  # PRO_f0 attacks CON_f0
        ("CON", 0, 0),  # CON_f0 attacks PRO_f0
    ]
    cases = [
        (s, af, tf, v)
        for (s, af, tf) in base_cases
        for v in variants
    ]

    print(f"총 (case × variant): {len(cases)} 건\n")

    if args.dry_run:
        for s, af, tf, v in cases:
            path = _cache_path(topic["id"], s, af, tf, v)
            print(f"  {path.name}  ← attacker={s} af{af}/tf{tf}/v{v}")
        return

    generated, skipped, failed = 0, 0, 0
    t_start = time.time()

    for attacker_stance, att_focus, tgt_focus, variant_idx in cases:
        path = _cache_path(topic["id"], attacker_stance, att_focus, tgt_focus, variant_idx)
        if args.skip_existing and path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("prompt_version") == CACHE_PROMPT_VERSION and existing.get("speech"):
                    skipped += 1
                    print(f"[SKIP] {path.name}")
                    continue
            except Exception:
                pass

        target_stance = "CON" if attacker_stance == "PRO" else "PRO"
        print(f"\n[{generated + 1}] {path.name}")
        print(f"  {attacker_stance}_f{att_focus}/v{variant_idx} attacks {target_stance}_f{tgt_focus}/v{variant_idx}")

        try:
            entry = _generate_one(topic, attacker_stance, att_focus, tgt_focus, variant_idx)
            path.write_text(json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  → 저장: {path.name}")
            generated += 1
        except Exception as e:
            import traceback
            print(f"  [실패] {e}")
            traceback.print_exc()
            failed += 1

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"완료: {generated}건 생성, {skipped}건 skip, {failed}건 실패, 소요 {elapsed:.0f}초")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
