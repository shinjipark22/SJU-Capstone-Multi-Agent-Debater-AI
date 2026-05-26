"""
AI 입론 캐시 사전 생성 스크립트.

지정된 (topic, stance, intensity, focus_idx) 조합에 대해 N개 variant 를 생성해
data/cache/openings/{topic_id}_{stance}_i{intensity}_f{focus_idx}.json 에 저장한다.

각 케이스 키 → variants 리스트 (랜덤 선택용).

사용:
  # 시범 — 단일 케이스만
  python scripts/cache/generate_opening_cache.py \
      --topic tech_003 --stance PRO --intensity 3 --focus 0 --variants 3

  # 토픽 전체 (해당 토픽의 모든 stance × intensity × focus)
  python scripts/cache/generate_opening_cache.py --topic tech_003 --variants 3

  # 이미 캐시된 케이스는 skip (resume 가능)
  python scripts/cache/generate_opening_cache.py --topic tech_003 --variants 3 --skip-existing
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


from src.phase0.persona_factory import INTENSITY_PROFILES, _build_system_prompt  # noqa: E402
from src.phase1.stage1_opening.nodes import _generate_opening  # noqa: E402


_TOPICS_PATH = _ROOT / "data" / "topics_20260323_processed.json"
_SEARCH_QUERIES_PATH = _ROOT / "data" / "search_queries.json"
_CACHE_DIR = _ROOT / "data" / "cache" / "openings"

# 캐시 버전 — 프롬프트 수정 시 올려서 무효화
CACHE_PROMPT_VERSION = "v1"


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


def _cache_path(topic_id: str, stance: str, intensity: int, focus_idx: int) -> Path:
    return _CACHE_DIR / f"{topic_id}_{stance}_i{intensity}_f{focus_idx}.json"


def _build_agent_for_cache(
    topic: dict, stance: str, intensity: int, focus_area: str
) -> dict:
    """캐시 생성용 가상 agent dict — 노드가 기대하는 필드만 채운다.

    AgentPersona 생성 없이 직접 dict 로 — focus_area 자동 할당 로직 우회.
    """
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


def _generate_one_case(
    topic: dict, stance: str, intensity: int, focus_idx: int, focus_area: str,
    variants: int,
) -> dict:
    """한 케이스에 대해 N variants 생성 후 cache entry 반환."""
    agent = _build_agent_for_cache(topic, stance, intensity, focus_area)
    name = _agent_name(stance)

    speeches = []
    tool_calls_total = []
    for n in range(variants):
        t0 = time.time()
        print(f"  variant {n+1}/{variants} 생성 중...", flush=True)
        speech, raw, tool_calls = _generate_opening(
            agent, topic["title"], stance, name, focus_area,
        )
        dt = time.time() - t0
        print(f"  variant {n+1}/{variants} 완료 ({dt:.1f}s, {len(speech)}자, search={len(tool_calls)})", flush=True)
        speeches.append(speech)
        tool_calls_total.append(len(tool_calls))

    return {
        "prompt_version": CACHE_PROMPT_VERSION,
        "topic_id": topic["id"],
        "stance": stance,
        "intensity": intensity,
        "focus_idx": focus_idx,
        "focus_area": focus_area,
        "agent_name": name,
        "variants": speeches,
        "search_calls_per_variant": tool_calls_total,
        "generated_at": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", required=True, help="토픽 ID (예: tech_003)")
    parser.add_argument("--stance", choices=["PRO", "CON"], help="특정 stance 만 (생략 시 둘 다)")
    parser.add_argument("--intensity", type=int, choices=[1, 2, 3, 4, 5], help="특정 intensity 만 (생략 시 1~5)")
    parser.add_argument("--focus", type=int, help="특정 focus_idx 만 (생략 시 전체)")
    parser.add_argument("--max-focus-idx", type=int, default=1, help="이 인덱스까지만 (1:1+2:2 면 1, 3:3 까지 가려면 2). --focus 지정 시 무시")
    parser.add_argument("--variants", type=int, default=3, help="케이스당 variant 수 (기본 3)")
    parser.add_argument("--skip-existing", action="store_true", help="이미 캐시된 케이스 skip")
    parser.add_argument("--dry-run", action="store_true", help="실제 생성 없이 케이스 목록만 출력")
    args = parser.parse_args()

    topic = _load_topic(args.topic)
    print(f"\n[캐시 생성] 토픽: {topic['id']} — {topic['title']}\n")

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    stances = [args.stance] if args.stance else ["PRO", "CON"]
    intensities = [args.intensity] if args.intensity else [1, 2, 3, 4, 5]

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
                    print(f"  [경고] {stance} focus_idx={focus_idx} 범위 초과 (max={len(focuses)-1}), skip")
                    continue
                cases.append((stance, intensity, focus_idx, focuses[focus_idx]))

    print(f"총 케이스: {len(cases)}, variant 수: {args.variants}, 총 생성: {len(cases) * args.variants}건\n")

    if args.dry_run:
        for stance, intensity, focus_idx, focus_area in cases:
            print(f"  {stance} / intensity={intensity} / focus={focus_idx} → {focus_area[:40]}")
        return

    skipped = 0
    generated = 0
    t_start = time.time()

    for stance, intensity, focus_idx, focus_area in cases:
        path = _cache_path(topic["id"], stance, intensity, focus_idx)
        if args.skip_existing and path.exists():
            try:
                existing = json.loads(path.read_text(encoding="utf-8"))
                if existing.get("prompt_version") == CACHE_PROMPT_VERSION and len(existing.get("variants", [])) >= args.variants:
                    skipped += 1
                    print(f"[SKIP] {path.name} (이미 v{CACHE_PROMPT_VERSION}, variants {len(existing['variants'])}개)")
                    continue
            except Exception:
                pass

        print(f"\n[{generated + 1}/{len(cases) - skipped}] {path.name}")
        print(f"  stance={stance} intensity={intensity} focus_idx={focus_idx} ({focus_area[:50]})")

        try:
            entry = _generate_one_case(
                topic, stance, intensity, focus_idx, focus_area, args.variants,
            )
            path.write_text(
                json.dumps(entry, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  → 저장: {path}")
            generated += 1
        except Exception as e:
            import traceback
            print(f"  [실패] {e}")
            traceback.print_exc()

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"완료: {generated}건 생성, {skipped}건 skip, 소요 {elapsed:.0f}초")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
