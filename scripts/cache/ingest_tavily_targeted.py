"""
ingest_tavily_targeted.py — 논점 타깃 웹 문서를 Tavily 로 검색해 Pinecone KB 에 적재.

진단 결과: tech_003 의 검색어 생성·retrieval 자체는 정상이나, KB 에
'논점에 정확히 꽂히는 논증급 문서'가 부족(주제는 맞고 논점은 빗나간 문서가 상위).
→ 각 진영 두 논거에 직접 근거가 되는 큐레이션 쿼리로 Tavily advanced 검색 후 upsert.

기존 인프라 재사용:
  - Tavily: search_depth="advanced" (llm.py 의 운영 검색과 동일 깊이)
  - 저장:   src.graph.vector_store.upsert_search_results (운영 retrieval 과 동일 경로)

사용:
  python scripts/cache/ingest_tavily_targeted.py --stance PRO            # PRO 적재
  python scripts/cache/ingest_tavily_targeted.py --stance CON            # CON 적재
  python scripts/cache/ingest_tavily_targeted.py --stance CON --dry-run  # 검색만
"""
from __future__ import annotations

import argparse
import os
import sys
import time
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

from tavily import TavilyClient  # noqa: E402
from src.graph.vector_store import upsert_search_results  # noqa: E402

_tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))

_EXCLUDE = [
    "blog.naver.com", "m.blog.naver.com",
    "tistory.com", "brunch.co.kr",
    "linkedin.com", "medium.com",
    "velog.io", "daum.net",
]

# ── tech_003 PRO — 윤리적 안전장치가 경쟁력 ─────────────────────────────────
# 논거1: 규제 시대에는 윤리 기준 자체가 경쟁력(시장 진입권)이 된다
# 논거2: 신뢰를 잃은 AI 는 시장에서 퇴출된다 / 윤리성 확보는 사업적 이점
QUERIES_PRO = {
    "arg1_regulation_as_moat": [
        "EU AI Act 고위험 AI 적합성 평가 시장 출시 요건",
        "EU AI Act 컴플라이언스 기업 대응 비용 경쟁력",
        "NIST AI Risk Management Framework 기업 도입 사례",
        "한국 AI 기본법 고영향 인공지능 사업자 의무",
        "OECD AI 원칙 신뢰할 수 있는 AI 거버넌스",
        "AI 규제 준수 시장 진입장벽 경쟁우위 분석",
        "AI 투명성 설명가능성 위험관리 인간감독 규제 통과",
    ],
    "arg2_trust_or_exit": [
        "에어캐나다 AI 챗봇 잘못된 정보 배상 판결",
        "AI 챗봇 허위정보 기업 소송 손해배상 사례",
        "AI 서비스 신뢰 상실 고객 이탈 계약 해지",
        "딥페이크 저작권 침해 AI 기업 소송 시장 퇴출",
        "책임있는 AI 윤리 인증 공공조달 글로벌 진출 사례",
        "AI 개인정보 유출 규제 제재 기업 신뢰 하락",
        "신뢰할 수 있는 AI 장기 계약 기업 고객 도입 이점",
    ],
    # arg2 보강 — 'AI 신뢰 실패 → 철회·중단·퇴출'의 구체 실명 사례
    "arg2_failure_cases": [
        "메타 갤럭티카 Galactica AI 출시 3일 만에 철회",
        "마이크로소프트 Tay 챗봇 혐오 발언 16시간 만에 중단",
        "아마존 AI 채용 시스템 여성 차별 편향 폐기",
        "구글 제미나이 Gemini 이미지 생성 오류 서비스 중단 사태",
        "AI 제품 출시 후 신뢰 문제로 철회 중단 사례",
        "이루다 챗봇 개인정보 혐오발언 서비스 중단 사례",
        "AI 신뢰 실패 브랜드 평판 훼손 주가 하락 사례",
    ],
}

# ── tech_003 CON — 압도적 모델 성능이 경쟁력 ────────────────────────────────
# f0: 고성능 AI 의 사용자 효용과 사회 보호 사례
# f1: AI 성능 우위와 국가 안보 확보 (swap 으로 f1 승격 — 토픽 정합)
QUERIES_CON = {
    "f0_high_perf_utility": [
        "고성능 AI 모델 사용자 생산성 향상 효용 사례",
        "프런티어 AI 모델 성능 향상 실제 활용 성과",
        "고성능 AI 의료 진단 재난 대응 사회 기여 사례",
        "최첨단 AI 모델 벤치마크 성능과 사용자 경험 개선",
    ],
    "f1_perf_and_security": [
        "AI 모델 성능 우위 시장 점유율 결정 사례",
        "프런티어 AI 기술 패권 국가 안보 경쟁",
        "미중 AI 기술 경쟁 모델 성능 군사 안보",
        "최고 성능 AI 모델 생태계 독점 선점 효과",
        "AI 반도체 컴퓨팅 파워 성능 경쟁력 우위",
        "고성능 거대언어모델 시장 지배력 빅테크 경쟁",
    ],
}

ALL = {"PRO": QUERIES_PRO, "CON": QUERIES_CON}


def main(stance: str, dry_run: bool = False) -> None:
    groups = ALL[stance]
    total_q = total_items = total_vec = 0
    for group, queries in groups.items():
        print(f"\n=== [{stance}] {group} — 쿼리 {len(queries)}개 ===")
        for q in queries:
            total_q += 1
            try:
                res = _tavily.search(
                    q, max_results=5, search_depth="advanced",
                    exclude_domains=_EXCLUDE,
                )
                items = res.get("results", [])
            except Exception as e:
                print(f"  [검색실패] {q!r}: {e}")
                continue
            items = [r for r in items if (r.get("url") and r.get("content"))]
            total_items += len(items)
            print(f"  [{len(items)}건] {q}")
            for r in items:
                print(f"        - {(r.get('title') or '')[:55]}  ({r.get('url','')[:50]})")
            if dry_run or not items:
                continue
            try:
                n = upsert_search_results(query=q, results=items)
                total_vec += n
                print(f"        ↳ upsert: {n} vectors")
            except Exception as e:
                print(f"        ↳ upsert 실패: {e}")
            time.sleep(0.5)

    print(f"\n=== 완료 [{stance}] === 쿼리 {total_q} / 문서 {total_items} / "
          f"{'(dry-run, upsert 안 함)' if dry_run else f'벡터 {total_vec} 적재'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stance", choices=["PRO", "CON"], required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    main(stance=args.stance, dry_run=args.dry_run)
