"""발언 캐시 품질 검사 — 생성 스크립트는 LLM 출력을 검증 없이 저장하므로 배포 전에 돌린다.

검출 항목:
  - 문자 반복 붕괴 (예: "효율AIAIAI")
  - 비어 있는 논거 헤더 ("**논거 2: **")
  - 중국어·일본어 유출
  - 문장 중간에서 잘림 (종결 부호 없이 끝남)
  - 지나치게 짧은 본문
  - 입론 구조 누락 (논거 1 / 논거 2 / 결론)

사용:
  python scripts/cache/lint_cache.py                 # data/cache 전체
  python scripts/cache/lint_cache.py --topic tech_003
  python scripts/cache/lint_cache.py --delete-bad    # 문제 파일 삭제 (재생성용)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "data" / "cache"

RE_REPEAT = re.compile(r"(.{1,4}?)\1{4,}")                 # 같은 1~4자 조각이 5회 이상 연속
RE_EMPTY_HEADER = re.compile(r"\*\*[^*\n]{0,12}:\s*\*\*\s*$", re.M)
# 한자 2자 이상 연속, 가나, 중국어 문장부호. 한국 언론이 쓰는 단독 한자(美·中·日)는 허용.
RE_CJK = re.compile(r"[一-鿿]{2,}|[ぁ-ゟ゠-ヿ]|[。，]")
RE_LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")
RE_TERMINAL = re.compile(r"[.!?。」』)\]\*]\s*$|[다요죠야어지네][.!?]?\s*$")

OPENING_SECTIONS = ("논거 1", "논거 2", "결론")


def check(kind: str, text: str) -> list[str]:
    problems: list[str] = []
    stripped = text.strip()
    if len(stripped) < 200:
        problems.append(f"본문 짧음 ({len(stripped)}자)")
    if m := RE_REPEAT.search(stripped):
        problems.append(f"문자 반복 붕괴: …{stripped[max(0, m.start()-12):m.end()+4]!r}")
    if RE_EMPTY_HEADER.search(stripped):
        problems.append("비어 있는 논거 헤더")
    no_links = RE_LINK.sub("", stripped)   # 링크 제목(외부 기사 제목)은 검사 대상에서 제외
    if m := RE_CJK.search(no_links):
        problems.append(f"중국어/일본어 유출: {no_links[max(0, m.start()-15):m.end()+15]!r}")
    # 안내문은 본문 → "---" → 링크 목록 구조라, 본문 부분만 떼어 종결 부호를 검사한다
    body = stripped.split("**더 자세히")[0] if "**더 자세히" in stripped else stripped
    body = body.rstrip().rstrip("-").rstrip()
    if not RE_TERMINAL.search(body):
        problems.append(f"문장 중간 잘림: …{body[-40:]!r}")
    if kind in ("openings", "role_reversals"):
        missing = [s for s in OPENING_SECTIONS if s not in stripped]
        if missing:
            problems.append(f"입론 구조 누락: {missing}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic")
    ap.add_argument("--delete-bad", action="store_true")
    args = ap.parse_args()

    files = sorted(CACHE.glob("*/*.json"))
    if args.topic:
        files = [f for f in files if f.name.startswith(args.topic + "_")]
    if not files:
        print("검사할 캐시 파일이 없습니다.")
        return 0

    bad = 0
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            text = data.get("speech") or data.get("text") or ""
        except Exception as e:  # noqa: BLE001
            print(f"[BAD] {f.relative_to(CACHE)}: JSON 파싱 실패 ({e})")
            bad += 1
            continue
        problems = check(f.parent.name, text)
        if problems:
            bad += 1
            print(f"[BAD] {f.relative_to(CACHE)}")
            for p in problems:
                print(f"       - {p}")
            if args.delete_bad:
                f.unlink()
                print("       → 삭제 (재생성 필요)")
    print(f"\n검사 {len(files)}개 / 문제 {bad}개")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
