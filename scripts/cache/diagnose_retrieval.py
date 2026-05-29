"""
진단 — tech_003 PRO 입론 검색 체인 점검.

흐름: focus_area → (실제 Plan LLM) 검색어 생성 → Pinecone KB 조회(Tavily 0회)
각 쿼리가 KB에서 무엇을 어떤 score로 끌어오는지, 0.85 threshold 를 넘는지,
잡히는 문서가 '논증급'인지 '위키 일반론'인지 눈으로 보기 위한 덤프.

사용: python scripts/cache/diagnose_retrieval.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))


def _load_env() -> None:
    p = _ROOT / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_env()
os.environ["SPEECH_CACHE_ENABLED"] = "0"

import json  # noqa: E402
import src.phase1.stage1_opening.nodes as nodes  # noqa: E402
from src.phase0.persona_factory import _build_system_prompt  # noqa: E402
from src.graph.vector_store import semantic_cache_lookup, set_current_stance  # noqa: E402

TOPIC_ID = "tech_003"
STANCE = "PRO"
INTENSITY = 3
THRESHOLD = float(os.environ.get("PINECONE_THRESHOLD", "0.85"))
_TOPICS_PATH = _ROOT / "data" / "topics_20260323_processed.json"
_SQ_PATH = _ROOT / "data" / "search_queries.json"


def _load_topic():
    data = json.loads(_TOPICS_PATH.read_text(encoding="utf-8"))
    for cat in data.get("categories", {}).values():
        for t in cat:
            if t["id"] == TOPIC_ID:
                return t
    raise SystemExit("topic not found")


def _focus_areas():
    return json.loads(_SQ_PATH.read_text(encoding="utf-8"))[TOPIC_ID][STANCE]


def _kind(url: str) -> str:
    return "WIKI" if "wikipedia.org" in (url or "") else "WEB "


def main():
    set_current_stance(None)  # URL 제외 비활성 — 순수 조회
    topic = _load_topic()
    focuses = _focus_areas()

    for fidx in (0, 1):
        focus_area = focuses[fidx]
        print("\n" + "=" * 90)
        print(f"[f{fidx}] focus_area = {focus_area}")
        print("=" * 90)

        agent = {
            "agent_id": "diag",
            "stance": STANCE,
            "intensity": INTENSITY,
            "system_prompt": _build_system_prompt(
                agent_id="diag", stance=STANCE, intensity=INTENSITY,
                title=topic["title"], pro=topic["pro"], con=topic["con"],
                description=topic.get("description_long"),
            ),
            "focus_area": focus_area,
        }

        # 실제 Plan LLM 으로 검색어 생성
        plan, _ = nodes._generate_plan(agent, topic["title"], STANCE, focus_area)
        outline = plan.get("argument_outline", [])
        queries = plan.get("search_queries", [])
        print("\n[Plan 이 생성한 논거 outline]")
        for i, o in enumerate(outline, 1):
            print(f"  논거{i}: {o}")
        print("\n[Plan 이 생성한 검색어]")
        for q in queries:
            print(f"  - {q}")

        # 각 쿼리로 KB 조회 (threshold 0 으로 전부 본다)
        for q in queries:
            print(f"\n  ── 쿼리: {q!r}")
            hits = semantic_cache_lookup(q, top_k=15, threshold=0.0, max_urls=5)
            if not hits:
                print("     (KB 조회 결과 없음 — Pinecone 빈 응답)")
                continue
            for h in hits:
                passed = "✅PASS" if h["score"] >= THRESHOLD else "  miss"
                snippet = (h["content"] or "").replace("\n", " ")[:90]
                print(f"     {passed} score={h['score']:.3f} {_kind(h['url'])} {h['title'][:40]}")
                print(f"            ↳ {snippet}")

    print(f"\n(threshold = {THRESHOLD})")


if __name__ == "__main__":
    main()
