"""auto_test 결과를 입력으로 어시스턴트 5단계 가이드를 생성하는 E2E.

기존 assistant_guide_e2e.py 는 토픽별 하드코딩 preset 을 쓰지만, 이건
실제 자동 테스트 산출물 (test_results/auto_test_2v2_env_002_*.txt) 의
실제 발언들을 사용해서 사용자가 각 단계에서 발언 직전 상태의
GuideContext 를 페이즈별로 구성한다.

사용:
    python tests/assistant_guide_from_autotest.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
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


from src.debate_assistant import (  # noqa: E402
    GuideContext,
    HistoryEntry,
    build_guide_message,
    get_user_slot_focus_area,
)


# ─────────────────────────────────────────────────────────────────────────
# env_002 auto_test 결과에서 추출한 실제 발언
# 원본: test_results/auto_test_2v2_env_002_20260412_110206.txt
# 사용자 진영: PRO ("에너지 정책의 책임이다")
# ─────────────────────────────────────────────────────────────────────────

TOPIC = "인플레이션의 책임은 기후변화가 아닌 에너지 정책에 있다"
TOPIC_ID = "env_002"
USER_STANCE = "PRO"
PRO_CLAIM = "에너지 정책의 책임이다"
CON_CLAIM = "기후변화의 책임이다"


# (speaker_id, stance, phase, content)
OPENINGS = [
    ("agent_3", "PRO", "opening",
     "찬성1 입니다. 저는 인플레이션의 책임이 기후변화보다는 에너지 정책에 있다는 입장을 취합니다. "
     "유럽 연합의 Green Deal 이후 가스 가격 100% 이상 상승, 미국의 파리협약 탈퇴와 규제 완화로 인한 "
     "석유 생산량 증가가 인플레이션을 가속화시킨 주요 요인입니다."),
    ("agent_1", "CON", "opening",
     "반대1 입니다. 인플레이션의 책임은 에너지 정책보다는 기후변화에 더 큰 영향을 받습니다. "
     "기후변화로 인한 폭염·가뭄으로 농작물 생산량이 감소하고 가격이 상승했으며, "
     "해충·질병 확산도 농작물 피해를 증가시켜 인플레이션의 직접 원인이 됩니다."),
    ("agent_2", "CON", "opening",
     "반대2 입니다. 저는 인플레이션의 책임이 에너지 정책이 아닌 기후변화에 있다는 입장입니다. "
     "2021년 유럽 가뭄으로 소맥·옥수수 작황 감소, 유럽연합 식품 가격이 7~9월 사이 3.4%→4.2% 상승. "
     "독일 라인강 수위 저하로 인한 발전소 냉각수 부족, 2022년 6월 전기가격 38.4% 상승 — "
     "기후변화가 공급망과 에너지 생산을 동시에 압박하고 있습니다."),
]

USER_OPENING = (
    "저는 이 논제에 찬성합니다. 인플레이션의 책임은 기후변화가 아닌 에너지 정책에 있다고 봅니다. "
    "2021~2023년 인플레이션의 주요 동인은 에너지 가격 급등이며, 러시아-우크라이나 전쟁과 OPEC+ 감산 등 "
    "에너지 정책·지정학적 요인이 직접 원인입니다. 또한 유럽의 급진적 탈탄소 정책(석탄 조기 폐쇄, 원전 폐지)은 "
    "에너지 공급 여력을 줄여 가격 변동성을 키웠습니다."
)

CHAINED = [
    ("agent_1", "CON", "chained_rebuttal",
     "상대방의 주장은 재생에너지 부족과 가스·석유 가격 상승만으로 인플레이션을 단정짓는 문제가 있습니다. "
     "인플레이션은 공급과 수요, 통화 정책 등 다양한 요인에 의해 결정되며, 단순히 에너지 가격 상승만으로 "
     "인플레이션을 설명하는 것은 오류입니다."),
    ("agent_3", "PRO", "chained_rebuttal",
     "상대방의 주장 중 농작물 피해와 가격 상승의 인과관계는 다수의 다른 요인들 때문에 직접적 귀결로 보기 어렵습니다. "
     "2019년 미국 콩 생산량 감소는 콩 벼룩 확산일 수도 있지만, 단순히 기후변화 때문이라고 단정 짓기 어렵습니다."),
    ("agent_2", "CON", "chained_rebuttal",
     "상대방의 주장 중 에너지 정책만을 문제로 지목하는 단순화된 접근은 인플레이션의 복잡성을 왜곡합니다. "
     "글로벌 에너지 시장은 지정학적 요인, 공급망 문제, 기후변화 등 다양한 요소들의 복합적 상호작용에 의해 움직입니다."),
]

USER_CHAINED = (
    "기후변화가 식량 가격에 영향을 미친다는 점은 인정하지만, 같은 기후 조건에서도 에너지 정책이 다른 국가 간 "
    "인플레이션 격차가 크다는 것은 정책이 더 결정적이라는 증거입니다."
)

# 자유논박 — 사용자는 PRO 입장에서 방어/공격 모두 하는데, AI 1 의 첫 자유논박 발언을 받기 직전
FREE_OPP_ATTACK_USER = (
    "상대방은 에너지 가격 급등이 인플레이션의 주요 원인이라고 주장하며 에너지 정책의 실패로 귀결짓고 있습니다. "
    "하지만 이 주장은 기후변화가 장기적으로 에너지 공급과 수요에 미치는 영향을 무시하고 있습니다. "
    "기후변화는 재생에너지 생산의 불확실성을 증가시키고, 폭염·가뭄 등 극단적 기후는 에너지 수요를 증가시키며, "
    "이는 에너지 가격 상승을 더욱 가속화합니다."
)

# 역할반전 — agent_2 가 사용자 진영(PRO) 역으로 발언
ROLE_REVERSAL_OPP = (
    "### 논거 1: 에너지 가격 급등의 직접적 원인\n"
    "2021~2023년 인플레이션의 주요 동인은 에너지 가격 급등이며, 러시아-우크라이나 전쟁과 OPEC+ 감산 등 "
    "에너지 정책·지정학적 요인이 직접 원인입니다. 기후변화는 간접적 영향에 불과합니다.\n\n"
    "### 논거 2: 유럽의 급진적 탈탄소 정책\n"
    "유럽의 급진적 탈탄소 정책(석탄발전 조기 폐쇄, 원전 폐지)은 에너지 공급 여력을 줄여 가격 변동성을 키웠습니다.\n\n"
    "### 결론\n"
    "인플레이션의 책임은 에너지 정책에 있으며, 특히 지정학적 요인과 탈탄소 정책의 실패가 주된 원인입니다."
)

# 종합 — 사용자 첫 종합 직전, 가장 최근 AI 발언
SYNTHESIS_PRIOR = [
    ("agent_3", "PRO", "synthesis",
     "기후변화의 영향은 분명히 존재하지만, 에너지 정책의 실패가 최근 인플레이션의 주요 원인임을 부인하기 어렵습니다. "
     "특히 지정학적 요인과 탈탄소 정책의 부적절한 실행이 가격 상승을 가속화했습니다."),
    ("agent_1", "CON", "synthesis",
     "인플레이션의 원인을 에너지 정책으로만 귀결짓는 것은 곤란합니다. 기후변화로 인한 자연 재해는 농업 생산과 "
     "인프라에 지속적인 피해를 준다는 점을 고려해야 합니다."),
    ("agent_2", "CON", "synthesis",
     "장기적으로 보았을 때 기후변화가 인플레이션에 미치는 영향은 점점 커질 것입니다. 에너지 정책의 실패는 "
     "단기적 효과가 뚜렷하지만, 기후변화는 장기적으로 지속되는 문제입니다."),
]


# ─────────────────────────────────────────────────────────────────────────
# 페이즈별 GuideContext 빌더
# ─────────────────────────────────────────────────────────────────────────

def _h(entries: list) -> list[HistoryEntry]:
    return [HistoryEntry(*e) for e in entries]


def _make_ctx(history: list, opponent_speech: str, *, focus_area: str | None = None) -> GuideContext:
    fa = focus_area if focus_area is not None else _USER_FOCUS_AREA
    return GuideContext(
        topic=TOPIC,
        user_stance=USER_STANCE,
        pro_claim=PRO_CLAIM,
        con_claim=CON_CLAIM,
        topic_id=TOPIC_ID,
        user_focus_area=fa,
        assistant_name="비비드",
        opponent_speech=opponent_speech,
        history=history,
        links=[],
    )


def make_ctx_for(phase: str) -> GuideContext:
    """auto_test 데이터에서 해당 phase의 '사용자 발언 직전' 상태를 재구성."""
    if phase == "opening":
        # 사용자 입론 직전. 가장 최근 상대 발언 = agent_2 CON 입론.
        return _make_ctx(history=_h(OPENINGS), opponent_speech=OPENINGS[-1][3])

    if phase == "chained_rebuttal":
        # 사용자 연쇄논박 직전. 사용자 입론 + 모든 AI 입론·연쇄논박 끝난 상태.
        hist = _h(OPENINGS) + [HistoryEntry("user", "PRO", "opening", USER_OPENING)] + _h(CHAINED)
        return _make_ctx(history=hist, opponent_speech=CHAINED[-1][3])

    if phase == "free_rebuttal":
        # 사용자 자유논박 직전. 모든 연쇄논박까지 끝나고 AI 1 의 첫 공격이 들어온 상태.
        hist = (
            _h(OPENINGS)
            + [HistoryEntry("user", "PRO", "opening", USER_OPENING)]
            + _h(CHAINED)
            + [HistoryEntry("user", "PRO", "chained_rebuttal", USER_CHAINED)]
            + [HistoryEntry("agent_1", "CON", "free_rebuttal", FREE_OPP_ATTACK_USER)]
        )
        return _make_ctx(history=hist, opponent_speech=FREE_OPP_ATTACK_USER)

    if phase == "role_reversal":
        # 사용자 역할반전(CON) 발언 직전. agent_2 가 PRO 역으로 발언한 상태.
        # focus_area 는 역할반전이라 reversed_stance 기반 — 함수가 알아서 처리하므로 일반 focus 그대로 전달.
        hist = (
            _h(OPENINGS)
            + [HistoryEntry("user", "PRO", "opening", USER_OPENING)]
            + _h(CHAINED)
            + [HistoryEntry("user", "PRO", "chained_rebuttal", USER_CHAINED)]
            + [HistoryEntry("agent_1", "CON", "free_rebuttal", FREE_OPP_ATTACK_USER)]
            + [HistoryEntry("agent_2", "PRO", "role_reversal", ROLE_REVERSAL_OPP)]  # agent_2 가 진영 반전
        )
        return _make_ctx(history=hist, opponent_speech=ROLE_REVERSAL_OPP)

    if phase == "synthesis":
        # 사용자 종합 발언 직전. AI 3명이 종합 한 번씩 한 상태.
        hist = (
            _h(OPENINGS)
            + [HistoryEntry("user", "PRO", "opening", USER_OPENING)]
            + _h(CHAINED)
            + [HistoryEntry("user", "PRO", "chained_rebuttal", USER_CHAINED)]
            + _h(SYNTHESIS_PRIOR)
        )
        return _make_ctx(history=hist, opponent_speech=SYNTHESIS_PRIOR[-1][3])

    raise ValueError(f"unsupported phase: {phase}")


# ─────────────────────────────────────────────────────────────────────────

_USER_FOCUS_AREA = get_user_slot_focus_area(
    TOPIC_ID, USER_STANCE, excluded_focuses=["에너지 정책 실패 인플레이션 영향 분석"]
)
print(f"[setup] topic={TOPIC_ID} stance={USER_STANCE} focus_area={_USER_FOCUS_AREA!r}")


_PHASES = ["opening", "chained_rebuttal", "free_rebuttal", "role_reversal", "synthesis"]


def main() -> int:
    out_dir = _ROOT / "test_results" / "assistant_guide"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results: dict = {}
    for phase in _PHASES:
        print(f"\n{'=' * 70}")
        print(f" [{phase}]")
        print(f"{'=' * 70}")
        try:
            ctx = make_ctx_for(phase)
            msg = build_guide_message(phase, ctx)
            results[phase] = {"status": "ok", "output": msg}
            print(msg)
        except Exception as e:
            results[phase] = {"status": "error", "error": str(e), "type": type(e).__name__}
            print(f"[ERROR] {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    md_path = out_dir / f"guide_from_autotest_env_002_{ts}.md"
    json_path = out_dir / f"guide_from_autotest_env_002_{ts}.json"

    with md_path.open("w", encoding="utf-8") as f:
        f.write(f"# DebateAssistant — env_002 auto_test 기반 E2E ({ts})\n\n")
        f.write(f"- 토론 주제: {TOPIC}\n")
        f.write(f"- 사용자 진영: {USER_STANCE}\n")
        f.write(f"- focus_area (입론): {_USER_FOCUS_AREA}\n\n")
        for phase, r in results.items():
            f.write(f"## {phase}\n\n")
            if r["status"] == "ok":
                f.write(r["output"])
                f.write("\n\n---\n\n")
            else:
                f.write(f"**ERROR** ({r['type']}): {r['error']}\n\n---\n\n")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "topic": TOPIC, "results": results},
                  f, ensure_ascii=False, indent=2)

    print(f"\n\n저장 →\n  {md_path}\n  {json_path}")
    fail = sum(1 for r in results.values() if r["status"] != "ok")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
