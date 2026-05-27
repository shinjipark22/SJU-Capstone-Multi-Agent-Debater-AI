"""
어시스턴트 역할반전 가이드 캐시 사전 생성 스크립트 (5번째 캐시 타입).

(topic, reversed_stance) 케이스당 variant 1/2/3 각각 1회씩 LLM 호출 →
data/cache/assistant_role_reversals/{topic}_{reversed_stance}__v{N}.json

text_guide.build_guide_message('role_reversal', ctx) 출력을 그대로 저장.
focus_area 미사용 (text_guide 의 role_reversal phase 가 focus_area 안 씀).

사용:
  python scripts/cache/generate_assistant_role_reversal_cache.py --topic tech_003
  python scripts/cache/generate_assistant_role_reversal_cache.py --topic tech_003 --variant-idx 2
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
_CACHE_DIR = _ROOT / "data" / "cache" / "assistant_role_reversals"

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


def _cache_path(topic_id: str, reversed_stance: str, variant_idx: int) -> Path:
    return _CACHE_DIR / f"{topic_id}_{reversed_stance}__v{variant_idx}.json"


def _generate_one(topic: dict, reversed_stance: str, variant_idx: int) -> dict:
    # text_guide 의 role_reversal phase 는 user_stance 의 '반대' 진영을 옹호하므로
    # 캐시할 reversed_stance 가 곧 옹호 대상. user_stance 는 반대로 박는다.
    user_stance = "CON" if reversed_stance == "PRO" else "PRO"

    ctx = GuideContext(
        topic=topic["title"],
        user_stance=user_stance,
        assistant_name="비비드",
        pro_claim=topic.get("pro", "찬성"),
        con_claim=topic.get("con", "반대"),
        topic_id=topic["id"],
        cache_variant_idx=None,  # 생성 중 캐시 hit 막기
    )

    t0 = time.time()
    out_search: dict = {}
    text = build_guide_message("role_reversal", ctx, out_search=out_search)
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
        "reversed_stance": reversed_stance,
        "variant_idx": variant_idx,
        "speech": text,
        "search_excerpts": excerpts,
        "generated_at": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True)
    parser.add_argument("--stance", choices=["PRO", "CON"], help="reversed_stance (생략 시 둘 다)")
    parser.add_argument("--variant-idx", type=int, choices=[1, 2, 3])
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    topic = _load_topic(args.topic)
    print(f"\n[어시스턴트 역할반전 캐시] 토픽: {topic['id']} — {topic['title']}\n")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    stances = [args.stance] if args.stance else ["PRO", "CON"]
    variants = [args.variant_idx] if args.variant_idx else list(ALL_VARIANTS)

    cases = [(s, v) for s in stances for v in variants]

    print(f"총 (case × variant): {len(cases)} 건\n")

    if args.dry_run:
        for s, v in cases:
            path = _cache_path(topic["id"], s, v)
            print(f"  {path.name}  ← reversed={s}/v{v}")
        return

    generated, skipped, failed = 0, 0, 0
    t_start = time.time()

    for reversed_stance, variant_idx in cases:
        path = _cache_path(topic["id"], reversed_stance, variant_idx)
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
        print(f"  reversed={reversed_stance} v{variant_idx}")

        try:
            entry = _generate_one(topic, reversed_stance, variant_idx)
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
