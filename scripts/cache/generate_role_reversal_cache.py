"""
역할반전 (AI) 캐시 사전 생성 스크립트.

(topic, reversed_stance, intensity) 케이스당 variant 1/2/3 각각 1회씩 LLM 호출 →
data/cache/role_reversals/{topic}_{reversed_stance}_i{intensity}__v{N}.json

설계 노트:
  - 역할반전은 실제 토론에선 'opponent_openings' (반전된 진영의 기존 입론) 를
    참고해 발언함. 캐시 생성 시점엔 그게 없으므로 **빈 참고** 로 둔다.
  - 결과: topic + search 기반 입론 톤 발언.

사용:
  python scripts/cache/generate_role_reversal_cache.py --topic tech_003
  python scripts/cache/generate_role_reversal_cache.py --topic tech_003 --variant-idx 2
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
from src.graph.llm import safe_search_invoke  # noqa: E402
from src.phase1.stage4_role_reversal.nodes import (  # noqa: E402
    _build_role_reversal_prompt,
    _generate_role_reversal,
)


_TOPICS_PATH = _ROOT / "data" / "topics_20260323_processed.json"
_CACHE_DIR = _ROOT / "data" / "cache" / "role_reversals"

CACHE_PROMPT_VERSION = "v1"
ALL_VARIANTS = (1, 2, 3)


def _load_topic(topic_id: str) -> dict:
    with _TOPICS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    for cat in data.get("categories", {}).values():
        for t in cat:
            if t["id"] == topic_id:
                return t
    raise SystemExit(f"topic_id '{topic_id}' not found")


def _cache_path(topic_id: str, reversed_stance: str, intensity: int, variant_idx: int) -> Path:
    return _CACHE_DIR / f"{topic_id}_{reversed_stance}_i{intensity}__v{variant_idx}.json"


def _build_agent(topic: dict, original_stance: str, intensity: int) -> dict:
    system_prompt = _build_system_prompt(
        agent_id="agent_cache_rr",
        stance=original_stance,
        intensity=intensity,
        title=topic["title"],
        pro=topic["pro"],
        con=topic["con"],
        description=topic.get("description_long"),
    )
    return {
        "agent_id": "agent_cache_rr",
        "stance": original_stance,
        "intensity": intensity,
        "system_prompt": system_prompt,
    }


def _agent_name(reversed_stance: str) -> str:
    return "찬성2" if reversed_stance == "PRO" else "반대2"


def _generate_one(
    topic: dict, reversed_stance: str, intensity: int, variant_idx: int,
    search_text: str, search_query: str,
) -> dict:
    original_stance = "CON" if reversed_stance == "PRO" else "PRO"
    agent = _build_agent(topic, original_stance, intensity)
    name = _agent_name(reversed_stance)

    prompt = _build_role_reversal_prompt(
        topic=topic["title"],
        reversed_stance=reversed_stance,
        agent_name=name,
        search_results=search_text,
        opponent_openings="(캐시 생성 — 토론 히스토리 없음. 토픽과 검색 자료 기반으로 자유롭게 입론하라)",
    )

    t0 = time.time()
    speech, raw, llm_tool_logs = _generate_role_reversal(agent, prompt)
    dt = time.time() - t0

    excerpts = []
    # 사전검색 자료 — variant 무관 공용
    if search_text:
        excerpts.append({"query": search_query, "content": search_text[:2000]})
    # LLM 이 추가로 호출한 tool 결과
    for tc in (llm_tool_logs or []):
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
        "reversed_stance": reversed_stance,
        "intensity": intensity,
        "variant_idx": variant_idx,
        "agent_name": name,
        "speech": speech,
        "search_excerpts": excerpts,
        "generated_at": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--stance", choices=["PRO", "CON"], help="reversed_stance (생략 시 둘 다)")
    parser.add_argument("--intensity", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--variant-idx", type=int, choices=[1, 2, 3])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    topic = _load_topic(args.topic)
    print(f"\n[역할반전 캐시] 토픽: {topic['id']} — {topic['title']}\n")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    stances = [args.stance] if args.stance else ["PRO", "CON"]
    intensities = [args.intensity] if args.intensity else [3]  # 기본 intensity 3
    variants = [args.variant_idx] if args.variant_idx else list(ALL_VARIANTS)

    cases = [
        (s, i, v) for s in stances for i in intensities for v in variants
    ]

    print(f"총 (case × variant): {len(cases)} 건\n")

    if args.dry_run:
        for s, i, v in cases:
            path = _cache_path(topic["id"], s, i, v)
            print(f"  {path.name}  ← reversed={s}/i{i}/v{v}")
        return

    # 사전 검색은 (topic, reversed_stance) 단위로 1회만 수행 — variant 마다 동일
    search_per_stance = {}
    search_query_per_stance = {}
    for s in stances:
        stance_kr = "찬성" if s == "PRO" else "반대"
        query = f"{topic['title']} {stance_kr} 논거 통계 자료"
        search_query_per_stance[s] = query
        print(f"[사전 검색] reversed={s}: {query[:60]}", flush=True)
        try:
            search_per_stance[s] = str(safe_search_invoke({"query": query}))[:2000]
        except Exception as e:
            print(f"  [경고] 검색 실패: {e}")
            search_per_stance[s] = ""

    generated, skipped, failed = 0, 0, 0
    t_start = time.time()

    for reversed_stance, intensity, variant_idx in cases:
        path = _cache_path(topic["id"], reversed_stance, intensity, variant_idx)
        if args.skip_existing and path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("prompt_version") == CACHE_PROMPT_VERSION and existing.get("speech"):
                    skipped += 1
                    print(f"[SKIP] {path.name}")
                    continue
            except Exception:
                pass

        print(f"\n[{generated + 1}] {path.name}")
        print(f"  reversed={reversed_stance} i{intensity} v{variant_idx}")

        try:
            entry = _generate_one(
                topic, reversed_stance, intensity, variant_idx,
                search_text=search_per_stance.get(reversed_stance, ""),
                search_query=search_query_per_stance.get(reversed_stance, ""),
            )
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
