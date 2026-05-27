"""
AI 입론 캐시 사전 생성 스크립트.

각 (topic, stance, intensity, focus_idx) 케이스에 대해 variant 1/2/3 각각 1회씩
LLM 호출하여 생성 → data/cache/openings/{topic}_{stance}_i{intensity}_f{focus}__v{N}.json
에 저장한다. 한 파일은 발언 1개 (`"speech": "..."`).

사용:
  # 토픽 전체 (모든 stance × intensity × focus 의 v1, v2, v3 모두)
  python scripts/cache/generate_opening_cache.py --topic tech_003

  # 단일 케이스만
  python scripts/cache/generate_opening_cache.py \
      --topic tech_003 --stance PRO --intensity 3 --focus 0

  # 특정 variant_idx 만 (재생성용)
  python scripts/cache/generate_opening_cache.py --topic tech_003 --variant-idx 2

  # dry-run — 케이스 목록만
  python scripts/cache/generate_opening_cache.py --topic tech_003 --dry-run

캐시 lookup 비활성화 (재귀 hit 방지) — 환경변수 SPEECH_CACHE_ENABLED=0 자동 설정.
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

# .env 로드 (TAVILY_API_KEY 등)
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
# 생성 중에는 캐시 lookup 끔 — 자기가 만든 v1 을 v2 생성에 잘못 쓰지 않게
os.environ["SPEECH_CACHE_ENABLED"] = "0"


from src.phase0.persona_factory import _build_system_prompt  # noqa: E402
from src.phase1.stage1_opening.nodes import _generate_opening  # noqa: E402


_TOPICS_PATH = _ROOT / "data" / "topics_20260323_processed.json"
_SEARCH_QUERIES_PATH = _ROOT / "data" / "search_queries.json"
_CACHE_DIR = _ROOT / "data" / "cache" / "openings"

# loader.py 의 EXPECTED_PROMPT_VERSION 과 일치 필수
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


def _cache_path(
    topic_id: str, stance: str, intensity: int, focus_idx: int, variant_idx: int,
) -> Path:
    return _CACHE_DIR / f"{topic_id}_{stance}_i{intensity}_f{focus_idx}__v{variant_idx}.json"


def _build_agent_for_cache(
    topic: dict, stance: str, intensity: int, focus_area: str
) -> dict:
    """캐시 생성용 가상 agent dict — 노드가 기대하는 필드만 채운다."""
    system_prompt = _build_system_prompt(
        agent_id="agent_cache",
        stance=stance,
        intensity=intensity,
        title=topic["title"],
        pro=topic["pro"],
        con=topic["con"],
        description=topic.get("description_long"),
    )
    return {
        "agent_id": "agent_cache",
        "stance": stance,
        "intensity": intensity,
        "system_prompt": system_prompt,
        "focus_area": focus_area,
    }


def _agent_name(stance: str) -> str:
    return "찬성1" if stance == "PRO" else "반대1"


def _excerpts_from_tool_calls(tool_calls: list, max_len: int = 2000) -> list:
    out = []
    for tc in tool_calls or []:
        if tc.get("name") != "search_web":
            continue
        query = (tc.get("args") or {}).get("query", "")
        result = str(tc.get("result", ""))
        out.append({"query": query, "content": result[:max_len]})
    return out


def _generate_one(
    topic: dict, stance: str, intensity: int, focus_idx: int, focus_area: str,
    variant_idx: int,
) -> dict:
    agent = _build_agent_for_cache(topic, stance, intensity, focus_area)
    name = _agent_name(stance)

    t0 = time.time()
    speech, raw, tool_calls = _generate_opening(
        agent, topic["title"], stance, name, focus_area,
    )
    dt = time.time() - t0
    excerpts = _excerpts_from_tool_calls(tool_calls)
    print(f"  완료 ({dt:.1f}s, {len(speech)}자, search={len(excerpts)})", flush=True)

    return {
        "prompt_version": CACHE_PROMPT_VERSION,
        "topic_id": topic["id"],
        "stance": stance,
        "intensity": intensity,
        "focus_idx": focus_idx,
        "focus_area": focus_area,
        "variant_idx": variant_idx,
        "agent_name": name,
        "speech": speech,
        "search_excerpts": excerpts,
        "generated_at": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="토픽 ID (예: tech_003)")
    parser.add_argument("--stance", choices=["PRO", "CON"], help="특정 stance 만")
    parser.add_argument("--intensity", type=int, choices=[1, 2, 3, 4, 5])
    parser.add_argument("--focus", type=int, help="특정 focus_idx 만")
    parser.add_argument("--max-focus-idx", type=int, default=1,
                        help="이 인덱스까지 (2:2 = 1, 1:1 = 0 도 포함되므로 1 권장). --focus 지정 시 무시")
    parser.add_argument("--variant-idx", type=int, choices=[1, 2, 3],
                        help="이 variant 만 (생략 시 1, 2, 3 모두)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="이미 존재하는 (case, variant) 파일은 skip")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    topic = _load_topic(args.topic)
    print(f"\n[캐시 생성] 토픽: {topic['id']} — {topic['title']}\n")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    stances = [args.stance] if args.stance else ["PRO", "CON"]
    intensities = [args.intensity] if args.intensity else [1, 2, 3, 4, 5]
    variants = [args.variant_idx] if args.variant_idx else list(ALL_VARIANTS)

    cases = []
    for stance in stances:
        focuses = _load_focus_areas(topic["id"], stance)
        if not focuses:
            print(f"  [경고] {topic['id']} {stance} focus_area 없음, skip")
            continue
        if args.focus is not None:
            focus_indices = [args.focus]
        else:
            max_idx = min(args.max_focus_idx, len(focuses) - 1)
            focus_indices = list(range(max_idx + 1))
        for intensity in intensities:
            for focus_idx in focus_indices:
                if focus_idx >= len(focuses):
                    continue
                for variant_idx in variants:
                    cases.append((stance, intensity, focus_idx, focuses[focus_idx], variant_idx))

    print(f"총 (case × variant): {len(cases)} 건\n")

    if args.dry_run:
        for stance, intensity, focus_idx, focus_area, variant_idx in cases:
            path = _cache_path(topic["id"], stance, intensity, focus_idx, variant_idx)
            print(f"  {path.name}  ← {stance}/i{intensity}/f{focus_idx}/v{variant_idx}  ({focus_area[:50]})")
        return

    generated, skipped, failed = 0, 0, 0
    t_start = time.time()

    for stance, intensity, focus_idx, focus_area, variant_idx in cases:
        path = _cache_path(topic["id"], stance, intensity, focus_idx, variant_idx)
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
        print(f"  stance={stance} i{intensity} f{focus_idx} v{variant_idx} ({focus_area[:50]})")

        try:
            entry = _generate_one(
                topic, stance, intensity, focus_idx, focus_area, variant_idx,
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
