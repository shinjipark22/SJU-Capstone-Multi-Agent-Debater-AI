"""DebateAssistant 안내문(text_guide) end-to-end 테스트.

토론 전체 파이프라인은 돌리지 않고, 단계별 GuideContext 만 손으로 구성해서
실제 Qwen2.5-32B-AWQ + tool calling + Pinecone/Tavily 와 통신.

결과는 콘솔 출력 + test_results/assistant_guide/guide_e2e_{ts}.md 저장.

사용:
    python tests/assistant_guide_e2e.py
    python tests/assistant_guide_e2e.py --phases opening chained_rebuttal
    python tests/assistant_guide_e2e.py --no-links     # 자동 링크 검색 끔
    python tests/assistant_guide_e2e.py --no-tools     # LLM tool calling 끔
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path


# repo 루트 import 가능하게
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def _load_env() -> None:
    """.env 파일에서 키들을 환경변수로 로드."""
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


from src.debate_assistant import (  # noqa: E402
    GuideContext,
    GuideLink,
    HistoryEntry,
    build_guide_message,
    get_user_slot_focus_area,
)


# 토픽별 시연용 컨텍스트 (history, opponent_speech 도 토픽에 맞게 구성)
_TOPIC_PRESETS = {
    "tech_001": {
        "title": "인공지능 발전은 인간 고용의 축소가 아닌 노동 시장의 재편이다",
        "pro": "노동 시장의 재편이다",
        "con": "인간 고용의 축소이다",
        "ai_used": ["산업혁명 자동화 기술 도입 이후 일자리 변화 역사적 사례"],
        "opponent_speech": (
            "AI 도입으로 빅테크를 중심으로 대규모 감원이 잇따르고 있고, "
            "골드만삭스는 전 세계 일자리의 약 25퍼센트가 자동화 위험에 노출되어 있다고 분석합니다. "
            "재교육·전환이라는 말은 듣기엔 좋지만, 실제로 사라지는 일자리의 속도와 규모를 "
            "따라가지 못하는 경우가 많습니다. 결국 노동시장의 '재편'이 아니라 '축소' 입니다."
        ),
        "history": [
            ("agent_1", "CON", "opening",
             "AI 도입으로 단순 사무직·콜센터·번역 분야에서 인력이 빠르게 감축되고 있습니다. 이는 노동시장 재편이라기보다 실질적인 고용 축소에 가깝습니다."),
            ("user", "PRO", "opening",
             "재교육과 직업 전환 프로그램으로 직원들이 새로운 역량을 갖추고 이동하고 있습니다. AI는 일자리를 단순히 줄이는 것이 아니라 노동시장을 재편하고 있습니다."),
            ("agent_1", "CON", "chained_rebuttal",
             "재교육 프로그램이 실제로 광범위하게 작동한다는 증거가 부족합니다. 골드만삭스의 25퍼센트 자동화 위험치는 무시할 수 없는 규모입니다."),
            ("user", "PRO", "chained_rebuttal",
             "25퍼센트는 위험 노출이지 실제 사라진 일자리가 아닙니다. 프롬프트 엔지니어·AI 윤리 감사관 같은 새 직종이 함께 등장하며 시장이 재편되고 있습니다."),
        ],
    },
    "env_002": {
        "title": "인플레이션의 책임은 기후변화가 아닌 에너지 정책에 있다",
        "pro": "에너지 정책의 책임이다",
        "con": "기후변화의 책임이다",
        "ai_used": ["에너지 정책 실패 인플레이션 영향 분석"],
        "opponent_speech": (
            "최근 이상 기후가 잦아지면서 농산물 수확이 불안정해지고 공급망에 차질이 생기고 있습니다. "
            "곡물·식료품 가격이 오르는 직접 원인이 기후변화 자체에 있어요. "
            "에너지 정책은 정부가 통제할 수 있지만 기후는 통제 밖이라, 인플레이션의 근본 원인은 기후변화입니다."
        ),
        "history": [
            ("agent_1", "CON", "opening",
             "기후변화로 인한 농산물 가격 상승과 공급망 차질이 글로벌 인플레이션의 핵심 동력입니다. 이는 정책 변수로 통제하기 어려운 구조적 원인입니다."),
            ("user", "PRO", "opening",
             "에너지 가격은 인플레이션의 가장 직접적 변수이며, 정부의 에너지 세제·보조금·재생에너지 전환 속도 같은 정책 선택이 인플레이션을 키웠습니다. 책임은 정책에 있습니다."),
            ("agent_1", "CON", "chained_rebuttal",
             "에너지 가격 변동도 결국 기후변화 대응 비용에서 시작합니다. 정책은 그 결과를 흡수할 뿐 원인이 아닙니다."),
            ("user", "PRO", "chained_rebuttal",
             "정책이 미리 안정적 에너지 전환을 설계했더라면 에너지 가격 충격은 훨씬 작았을 것입니다. 시점과 규모를 결정한 건 정책이지 기후가 아닙니다."),
        ],
    },
}


import argparse as _argparse

# --topic 인자만 빠르게 파싱 (DEFAULT_CTX 생성 전에)
_pre = _argparse.ArgumentParser(add_help=False)
_pre.add_argument("--topic-id", type=str, default="tech_001")
_pre.add_argument("--user-stance", type=str, default="PRO", choices=["PRO", "CON"])
_pre_args, _ = _pre.parse_known_args()

_TOPIC_ID = _pre_args.topic_id
_USER_STANCE = _pre_args.user_stance
_PRESET = _TOPIC_PRESETS.get(_TOPIC_ID)
if _PRESET is None:
    raise SystemExit(f"unsupported --topic-id: {_TOPIC_ID}. supported: {list(_TOPIC_PRESETS.keys())}")

_USER_FOCUS_AREA = get_user_slot_focus_area(
    _TOPIC_ID, _USER_STANCE, excluded_focuses=_PRESET["ai_used"]
)
print(f"[setup] topic={_TOPIC_ID} stance={_USER_STANCE} focus_area={_USER_FOCUS_AREA!r}")


DEFAULT_CTX = GuideContext(
    topic=_PRESET["title"],
    user_stance=_USER_STANCE,
    pro_claim=_PRESET["pro"],
    con_claim=_PRESET["con"],
    topic_id=_TOPIC_ID,
    user_focus_area=_USER_FOCUS_AREA,
    assistant_name="비비드",
    opponent_speech=_PRESET["opponent_speech"],
    history=[HistoryEntry(*h) for h in _PRESET["history"]],
    # ctx.links 는 비워둠 — 자동 검색 fallback 경로 테스트
    links=[],
)


_PHASES_ALL = ["opening", "chained_rebuttal", "free_rebuttal", "role_reversal", "synthesis"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phases",
        nargs="*",
        default=_PHASES_ALL,
        choices=_PHASES_ALL,
        help="실행할 단계 (기본: 전체 5단계)",
    )
    parser.add_argument(
        "--no-links",
        action="store_true",
        help="자동 링크 검색 비활성 (placeholder 표시)",
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="LLM tool calling 비활성 (순수 LLM 응답만)",
    )
    parser.add_argument(
        "--manual-link",
        action="store_true",
        help="ctx.links 에 수동 링크 1개 주입 (manual override 동작 확인)",
    )
    args = parser.parse_args()

    ctx = DEFAULT_CTX
    if args.manual_link:
        ctx = GuideContext(
            **{
                **DEFAULT_CTX.__dict__,
                "links": [
                    GuideLink(
                        title="OECD AI Outlook 2024 (수동 추가)",
                        url="https://www.oecd.org/digital/ai/",
                    )
                ],
            }
        )

    # tool calling 끄려면 phase 별 기본 LLMCall 을 우회
    custom_llm = None
    if args.no_tools:
        from src.debate_assistant.llm_adapter import make_llm_call

        custom_llm = make_llm_call(use_tools=False, label="assistant_e2e_notools")

    # 자동 링크 검색 끄려면 placeholder 가 박히도록 links 검색 함수 임시 비활성
    if args.no_links and not args.manual_link:
        # ctx.links 가 비어 있으면 자동 검색이 동작하므로, fetch 를 빈 결과로 monkey-patch
        from src.debate_assistant import links as _links_mod

        _links_mod.fetch_topic_links = lambda topic, stance, n=3: []

    out_dir = _ROOT / "test_results" / "assistant_guide"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results: dict = {}
    for phase in args.phases:
        print(f"\n{'=' * 70}")
        print(f" [{phase}]")
        print(f"{'=' * 70}")
        try:
            msg = build_guide_message(phase, ctx, llm=custom_llm)
            results[phase] = {"status": "ok", "output": msg}
            print(msg)
        except Exception as e:
            results[phase] = {"status": "error", "error": str(e), "type": type(e).__name__}
            print(f"[ERROR] {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    # 저장
    md_path = out_dir / f"guide_e2e_{ts}.md"
    json_path = out_dir / f"guide_e2e_{ts}.json"

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# DebateAssistant E2E — {ts}\n\n")
        f.write(f"- 토론 주제: {ctx.topic}\n")
        f.write(f"- 사용자 진영: {ctx.user_stance}\n")
        f.write(f"- 옵션: {vars(args)}\n\n")
        for phase, r in results.items():
            f.write(f"## {phase}\n\n")
            if r["status"] == "ok":
                f.write(r["output"])
                f.write("\n\n---\n\n")
            else:
                f.write(f"**ERROR** ({r['type']}): {r['error']}\n\n---\n\n")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": ts,
                "topic": ctx.topic,
                "user_stance": ctx.user_stance,
                "args": vars(args),
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\n\n저장 →\n  {md_path}\n  {json_path}")
    fail = sum(1 for r in results.values() if r["status"] != "ok")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
