"""
캐시 발언의 구체 인용이 search_excerpts 에 실제 있는지 자동 대조.

speech 의 본문(LINK 섹션·URL 제거 후)에서 '의심 토큰'을 추출하고
같은 토큰이 search_excerpts 합본에 있는지 확인.

ai_rebuttal 의 경우 자체 search 가 없으므로 같은 variant 의 attacker/target
opening 의 search_excerpts 를 추가로 본다.

매칭 안 되는 토큰 = hallucination 후보.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
_CACHE = _ROOT / "data" / "cache"
_OPENING_DIR = _CACHE / "openings"

# 고유명사·기관·인물 — speech 본문에 등장하면 search 에도 있어야
ENTITY_TOKENS = [
    "IBM", "Microsoft", "Google", "OpenAI", "NVIDIA", "AWS", "Gartner", "MIT",
    "Meta", "페이스북", "판도라보츠", "한화", "AIDA", "이루다", "연애의 과학",
    "Gemini", "AlphaGo", "GPT", "BERT", "Deep Think", "한국타이어",
    "MMMU", "GPQA", "SWE", "AI 안전 정상회담", "EU AI 법", "캘리포니아",
    "쇠렌 디네센", "외스테르고르", "MIT Technology Review",
    "디트로이트", "안면 인식", "딥페이크", "챗봇 정신병", "Chatbot psychosis",
]

YEAR_RE = re.compile(r"(20\d{2})년")
PCT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
DOLLAR_RE = re.compile(r"(\d+(?:,\d{3})*)\s*억\s*달러")

# speech 본문만 남기기 위한 클리너
_LINKS_HEADER_RE = re.compile(r"\n*\*\*더 자세히 보고 싶다면.*?\*\*\s*", re.DOTALL)
_MD_URL_RE = re.compile(r"\(https?://[^)\s]+\)")
_BARE_URL_RE = re.compile(r"https?://[^\s)]+")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")  # [text](url) — text 만 남김


def _clean_speech(speech: str) -> str:
    """speech 에서 LINK 섹션과 URL 을 떼어낸 본문만 반환."""
    # 마지막 LINK 섹션 잘라내기 — 어시스턴트 템플릿에서 ---뒤 마지막에 박힘
    m = _LINKS_HEADER_RE.search(speech)
    if m:
        speech = speech[:m.start()]
    # 마지막 ---로 끊긴 섹션이 링크만이면 잘라내기
    parts = speech.rsplit("\n---\n", 1)
    if len(parts) == 2 and _MD_LINK_RE.search(parts[1]) and not parts[1].strip().startswith("###"):
        # ---뒤 섹션이 링크 위주면 떼어냄
        body, tail = parts
        if len(_MD_LINK_RE.findall(tail)) >= 2:
            speech = body
    # 본문 안에 남은 마크다운 링크 URL 제거 (text만 남김)
    speech = _MD_LINK_RE.sub(r"\1", speech)
    speech = _BARE_URL_RE.sub("", speech)
    return speech


def _excerpt_text(data: dict) -> str:
    parts = []
    for ex in data.get("search_excerpts", []) or []:
        parts.append(ex.get("content", ""))
    return "\n".join(parts)


def _opening_excerpts_for_variant(
    topic_id: str, variant_idx: int, intensity: int = 3,
) -> str:
    """주어진 variant 의 모든 opening(PRO/CON × focus_idx) excerpt 합본."""
    parts = []
    for f in _OPENING_DIR.glob(f"{topic_id}_*_i{intensity}_*__v{variant_idx}.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            parts.append(_excerpt_text(d))
        except Exception:
            pass
    return "\n".join(parts)


def _normalize(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def audit_file(path: Path, type_name: str) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    speech = data.get("speech", "")
    if not speech:
        return [("ERROR", "speech 빈 값", "")]

    body = _clean_speech(speech)
    excerpt = _excerpt_text(data)

    # ai_rebuttal 은 자체 search 없으니 대응 variant 의 opening 합본 추가
    if type_name == "ai_rebuttals":
        variant_idx = data.get("variant_idx")
        topic_id = data.get("topic_id", "")
        if variant_idx:
            excerpt += "\n" + _opening_excerpts_for_variant(topic_id, variant_idx)

    if not excerpt.strip():
        return [("WARN", "search_excerpts 비어있음 (관련 cross-ref 도 비어 있음)", "")]

    body_norm = _normalize(body)
    excerpt_norm = _normalize(excerpt)

    misses = []

    for tok in ENTITY_TOKENS:
        tn = _normalize(tok)
        if tn in body_norm and tn not in excerpt_norm:
            idx = body.find(tok)
            ctx = body[max(0, idx - 25): idx + len(tok) + 35]
            misses.append(("ENTITY", tok, ctx.replace("\n", " ")))

    for m in YEAR_RE.finditer(body):
        year = m.group(1)
        if year not in excerpt:
            idx = m.start()
            ctx = body[max(0, idx - 25): idx + 35]
            misses.append(("YEAR", f"{year}년", ctx.replace("\n", " ")))

    for m in PCT_RE.finditer(body):
        pct = m.group(0)
        if _normalize(pct) not in excerpt_norm:
            idx = m.start()
            ctx = body[max(0, idx - 25): idx + 35]
            misses.append(("PCT", pct, ctx.replace("\n", " ")))

    for m in DOLLAR_RE.finditer(body):
        s = m.group(0)
        if _normalize(s) not in excerpt_norm:
            idx = m.start()
            ctx = body[max(0, idx - 25): idx + 35]
            misses.append(("DOLLAR", s, ctx.replace("\n", " ")))

    return misses


def main():
    by_type = {}
    for sub in sorted(_CACHE.iterdir()):
        if not sub.is_dir() or sub.name.startswith("_"):
            continue
        by_type[sub.name] = sorted(sub.glob("*.json"))

    total = sum(len(v) for v in by_type.values())
    flagged = 0
    print(f"\n[캐시 인용 검수] 총 {total} 파일 (LINK·URL 제외, ai_rebuttal 은 opening 합본 cross-ref)\n")

    for type_name, files in by_type.items():
        type_flags = []
        for f in files:
            misses = audit_file(f, type_name)
            if misses:
                type_flags.append((f.name, misses))

        if not type_flags:
            print(f"  ✅ {type_name}: 모두 깨끗 ({len(files)}개)")
            continue

        print(f"\n  ── {type_name} ({len(type_flags)}/{len(files)} 파일에서 mismatch) ──")
        for name, misses in type_flags:
            print(f"    [{name}]")
            for kind, tok, ctx in misses:
                print(f"      {kind:6} {tok!r}")
                print(f"             └ {ctx}")
            flagged += 1

    print(f"\n  요약: {flagged}/{total} 파일에 mismatch 후보")


if __name__ == "__main__":
    main()
