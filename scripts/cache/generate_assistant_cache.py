"""
어시스턴트 입론 가이드 캐시 사전 생성 스크립트.

각 (topic, stance, focus_idx) 케이스에 대해 variant 1/2/3 각각 1회씩 LLM 호출 →
data/cache/assistant_openings/{topic}_{stance}_f{focus}__v{N}.json 에 저장.

사용:
  python scripts/cache/generate_assistant_cache.py --topic tech_003
  python scripts/cache/generate_assistant_cache.py --topic tech_003 --variant-idx 2
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
os.environ["SPEECH_CACHE_ENABLED"] = "0"  # 생성 중 캐시 lookup 비활성


from src.debate_assistant.text_guide import (  # noqa: E402
    GuideContext,
    build_guide_message,
)


_TOPICS_PATH = _ROOT / "data" / "topics_20260323_processed.json"
_SEARCH_QUERIES_PATH = _ROOT / "data" / "search_queries.json"
_CACHE_DIR = _ROOT / "data" / "cache" / "assistant_openings"

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


def _load_focus_areas(topic_id: str, stance: str) -> list:
    with _SEARCH_QUERIES_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    return data.get(topic_id, {}).get(stance, [])


def _cache_path(topic_id: str, stance: str, focus_idx: int, variant_idx: int) -> Path:
    return _CACHE_DIR / f"{topic_id}_{stance}_f{focus_idx}__v{variant_idx}.json"


def _generate_one(
    topic: dict, stance: str, focus_idx: int, focus_area: str, variant_idx: int,
) -> dict:
    ctx = GuideContext(
        topic=topic["title"],
        user_stance=stance,
        assistant_name="비비드",
        pro_claim=topic.get("pro", "찬성"),
        con_claim=topic.get("con", "반대"),
        topic_id=topic["id"],
        user_focus_area=focus_area,
        cache_variant_idx=None,  # 생성 중이라 명시적으로 캐시 비활성
    )

    t0 = time.time()
    out_search: dict = {}
    text = build_guide_message("opening", ctx, out_search=out_search)
    dt = time.time() - t0
    print(f"  완료 ({dt:.1f}s, {len(text)}자, search_content={len(out_search.get('content',''))}자)", flush=True)

    excerpts = []
    if out_search.get("content"):
        excerpts.append({
            "query": out_search.get("query", ""),
            "content": out_search["content"][:2000],
        })

    return {
        "prompt_version": CACHE_PROMPT_VERSION,
        "topic_id": topic["id"],
        "stance": stance,
        "focus_idx": focus_idx,
        "focus_area": focus_area,
        "variant_idx": variant_idx,
        "speech": text,
        "search_excerpts": excerpts,
        "generated_at": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--stance", choices=["PRO", "CON"])
    parser.add_argument("--focus", type=int)
    parser.add_argument("--max-focus-idx", type=int, default=1)
    parser.add_argument("--variant-idx", type=int, choices=[1, 2, 3])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    topic = _load_topic(args.topic)
    print(f"\n[어시스턴트 입론 캐시] 토픽: {topic['id']} — {topic['title']}\n")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    stances = [args.stance] if args.stance else ["PRO", "CON"]
    variants = [args.variant_idx] if args.variant_idx else list(ALL_VARIANTS)

    cases = []
    for stance in stances:
        focuses = _load_focus_areas(topic["id"], stance)
        if not focuses:
            print(f"  [경고] {topic['id']} {stance} focus_area 없음")
            continue
        if args.focus is not None:
            focus_indices = [args.focus]
        else:
            max_idx = min(args.max_focus_idx, len(focuses) - 1)
            focus_indices = list(range(max_idx + 1))
        for focus_idx in focus_indices:
            if focus_idx >= len(focuses):
                continue
            for variant_idx in variants:
                cases.append((stance, focus_idx, focuses[focus_idx], variant_idx))

    print(f"총 (case × variant): {len(cases)} 건\n")

    if args.dry_run:
        for stance, focus_idx, focus_area, variant_idx in cases:
            path = _cache_path(topic["id"], stance, focus_idx, variant_idx)
            print(f"  {path.name}  ← {stance}/f{focus_idx}/v{variant_idx}  ({focus_area[:50]})")
        return

    generated, skipped, failed = 0, 0, 0
    t_start = time.time()

    for stance, focus_idx, focus_area, variant_idx in cases:
        path = _cache_path(topic["id"], stance, focus_idx, variant_idx)
        if args.skip_existing and path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("prompt_version") == CACHE_PROMPT_VERSION and existing.get("speech"):
                    skipped += 1
                    print(f"[SKIP] {path.name}")
                    continue
            except Exception:
                pass

        print(f"\n[{generated + 1}/{len(cases) - skipped}] {path.name}")
        print(f"  stance={stance} f{focus_idx} v{variant_idx} ({focus_area[:50]})")

        try:
            entry = _generate_one(topic, stance, focus_idx, focus_area, variant_idx)
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
