from __future__ import annotations
import json as pyjson
import re
from typing import Optional
import matplotlib.pyplot as plt

from .config import REBUTTAL_PHASES
from .models import TurnAnalysis, DimensionResult
from .memory import AnalysisMemory
from .prompts import SPEECH_SUMMARY_PROMPT, ARGUMENT_PROMPT, EVIDENCE_PROMPT, LANGUAGE_PROMPT
from .inference import qwen_chat

_PHASE_KR = {
    "opening": "입론",
    "chained_rebuttal": "연쇄 논박",
    "free_rebuttal": "자유 논박",
    "role_reversal": "역할 반전",
    "synthesis": "종합",
}


def _summarize_speech(speech_content: str) -> str:
    """발언 원문에서 핵심 주장/논거를 LLM으로 요약 추출."""
    raw = qwen_chat(SPEECH_SUMMARY_PROMPT, speech_content, max_new_tokens=150)
    try:
        data = _extract_json_object(raw)
        return str(data.get("summary", "요약 없음")).strip()
    except Exception:
        return speech_content[:200]


def _extract_evidence_from_text(content: str) -> list:
    """
    Python 정규식으로 현재 발언에서 출처 후보를 미리 추출.
    기관명 패턴 + 주변 수치/연도를 함께 뽑아서 LLM 환각 방지.
    """
    sources = []
    source_triggers = ["에 따르면", "보고서", "연구에", "연구소", "발표", "조사", "예측", "인용", "에 의하면"]
    sentences = re.split(r"(?<=[.!?])\s+|(?<=[다])\s+(?=[A-Z가-힣])", content)

    for sent in sentences:
        if not any(kw in sent for kw in source_triggers):
            continue
        org_match = re.search(
            r"([A-Za-z가-힣\s]{2,25}(?:연구소|연구원|기관|대학교?|포럼|은행|협회|위원회|연구))",
            sent,
        )
        abbr_match = re.search(r"([A-Z]{2,6})", sent)
        fig_match = re.search(r"\d+\.?\d*\s*(?:%|개|만\s*개|억|조|명|건|배)", sent)
        year_match = re.search(r"20\d{2}년?|\d{4}년", sent)

        org_name = None
        if org_match:
            org_name = org_match.group(1).strip()
        elif abbr_match:
            org_name = abbr_match.group(1).strip()

        sources.append({
            "인용_문장": sent.strip()[:120],
            "기관명": org_name,
            "수치": fig_match.group(0).strip() if fig_match else None,
            "연도": year_match.group(0) if year_match else None,
        })

    return sources


def _build_evidence_context(content: str, speaker_id: str, prev_speeches: list) -> str:
    """
    근거·출처 평가 전용 컨텍스트.
    python_추출_근거: Python이 현재 발언에서 뽑은 출처 후보 (환각 방지 앵커)
    """
    extracted = _extract_evidence_from_text(content)
    ctx = {
        "python_추출_근거": extracted if extracted else [],
        "현재_평가_대상": {
            "speaker_id": speaker_id,
            "content": content,
        },
        "이전_발언_흐름": [
            {
                "turn": item["turn_index"],
                "speaker": item["speaker_id"],
                "stance": item["speaker_stance"],
                "요약": item["summary"],
            }
            for item in prev_speeches
        ],
    }
    return pyjson.dumps(ctx, ensure_ascii=False, indent=2)


def _build_context(
    content: str,
    speaker_id: str,
    speaker_stance: str,
    turn_index: int,
    phase: str,
    target_id: Optional[str],
    speech_memory: list,
) -> str:
    """
    입론: 현재 발언만 평가 (이전 메모리 없음)
    반박: 이전 발언 요약(speech_memory)만 참고
    """
    phase_label = _PHASE_KR.get(phase, phase)
    is_rebuttal = phase in REBUTTAL_PHASES

    ctx = {
        "instruction": "현재 발화를 평가하세요.",
        "phase_rule": (
            "입론 단계입니다. 오직 현재 발언 자체만 평가하세요. 이전 발언은 참고하지 마세요."
            if not is_rebuttal
            else (
                f"[공격 턴] {speaker_id}가 {target_id}의 발언을 반박하고 있습니다. "
                f"speech_context에서 {target_id}의 발언을 찾아 "
                "상대방 논리의 허점을 얼마나 정확히 겨냥했는지 평가하세요."
                if target_id
                else "반박 단계입니다. speech_context에서 이전 발언들을 확인하고, "
                "상대방이 제기한 비판에 얼마나 잘 답했는지(방어) 평가하세요."
            )
        ),
        "current_speech": {
            "turn_index": turn_index,
            "speaker_id": speaker_id,
            "speaker_stance": speaker_stance,
            "phase": phase_label,
            "content": content,
            "target_id": target_id,
        },
        "speech_context": []
        if not is_rebuttal
        else [
            {
                "turn": item["turn_index"],
                "speaker": item["speaker_id"],
                "stance": item["speaker_stance"],
                "phase": item["phase"],
                "key_claims": item["summary"],
            }
            for item in speech_memory
        ],
    }
    return pyjson.dumps(ctx, ensure_ascii=False, indent=2)


def _extract_json_object(raw: str) -> dict:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("empty output")

    codeblock = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", raw)
    if codeblock:
        raw = codeblock.group(1).strip()

    decoder = pyjson.JSONDecoder()
    for idx, ch in enumerate(raw):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(raw[idx:])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    raise ValueError("no parseable json object found")


def _safe_parse_dimension(raw: str) -> dict:
    raw = (raw or "").strip()
    if raw and not raw.startswith("{"):
        raw = '{"score":' + raw

    try:
        data = _extract_json_object(raw)
    except Exception:
        score_match = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw or "")
        summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', raw or "")
        if score_match:
            data = {
                "score": float(score_match.group(1)),
                "summary": summary_match.group(1) if summary_match else "요약 없음",
            }
        else:
            print(f"  파싱 실패 | 원본: {str(raw)[:120]}")
            data = {"score": 5.0, "summary": "파싱 오류"}

    score = data.get("score", 5.0)
    try:
        score = float(score)
    except Exception:
        score = 5.0
    score = max(1.0, min(10.0, score))

    summary = str(data.get("summary", "요약 없음")).strip().replace("\n", " ")
    summary = re.sub(r"\s+", " ", summary)
    if not summary:
        summary = "요약 없음"

    return {"score": score, "summary": summary}


def _compose_overall_summary(argument: dict, source: dict, language: dict) -> str:
    return f"논증:{argument['summary']} | 근거:{source['summary']} | 언어:{language['summary']}"


def _compose_memory_summary(argument: dict, source: dict, language: dict, phase: str) -> str:
    prefix = "입론" if phase == "opening" else "논박"
    return f"{prefix}: {argument['summary']} / {source['summary']} / {language['summary']}"


def plot_live_debate(pro_percent: float, con_percent: float, title: str):
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.barh(["종합 우세 비율"], [con_percent], color="#E74C3C", label="반대")
    ax.barh(["종합 우세 비율"], [pro_percent], left=[con_percent], color="#2E86DE", label="찬성")
    ax.axvline(50, color="#222222", linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_xlabel("반대 (%)                |                찬성 (%)")
    ax.set_title(title)
    ax.legend(loc="upper center", ncol=2, frameon=False)
    ax.text(
        con_percent / 2, 0, f"반대 {con_percent:.1f}%",
        va="center", ha="center", color="white", fontsize=11,
    )
    ax.text(
        con_percent + pro_percent / 2, 0, f"찬성 {pro_percent:.1f}%",
        va="center", ha="center", color="white", fontsize=11,
    )
    plt.tight_layout()
    plt.show()


class DebatrixJudge:
    def __init__(
        self,
        topic: str,
        debate_format: str,
        user_id: str,
        user_stance: str,
        teams: dict,
    ):
        self.memory = AnalysisMemory(
            topic=topic,
            debate_format=debate_format,
            user_id=user_id,
            user_stance=user_stance,
            teams=teams,
        )
        self._turn = 0

    def judge_turn(
        self,
        speaker_id: str,
        speaker_stance: str,
        phase: str,
        speech_content: str,
        target_id: Optional[str] = None,
    ) -> TurnAnalysis:
        # 1. 발언 요약 → speech_memory에 먼저 저장
        print("  발언 요약 중...", end=" ")
        speech_summary = _summarize_speech(speech_content)
        print("완료")
        self.memory.add_speech(
            turn_index=self._turn,
            speaker_id=speaker_id,
            speaker_stance=speaker_stance,
            phase=phase,
            target_id=target_id,
            summary=speech_summary,
        )

        # 2. 컨텍스트 구성
        ctx_msg = _build_context(
            speech_content, speaker_id, speaker_stance,
            self._turn, phase, target_id,
            self.memory.speech_memory,
        )

        # 3. 3차원 평가
        print("  논증 심사 중...", end=" ")
        arg_raw = qwen_chat(ARGUMENT_PROMPT, ctx_msg, max_new_tokens=400, assistant_prefix='{"score":')
        print("완료")

        # 현재 턴 제외 — 자기 자신이 "이전 발언"으로 잡히는 버그 방지
        prev_speeches = [
            item for item in self.memory.speech_memory
            if item["turn_index"] < self._turn
        ]
        evidence_ctx_msg = _build_evidence_context(speech_content, speaker_id, prev_speeches)
        print("  근거 심사 중...", end=" ")
        src_raw = qwen_chat(EVIDENCE_PROMPT, evidence_ctx_msg, max_new_tokens=400, assistant_prefix='{"score":')
        print("완료")

        print("  언어 심사 중...", end=" ")
        lang_raw = qwen_chat(LANGUAGE_PROMPT, ctx_msg, max_new_tokens=400, assistant_prefix='{"score":')
        print("완료")

        argument = _safe_parse_dimension(arg_raw)
        evidence = _safe_parse_dimension(src_raw)
        language = _safe_parse_dimension(lang_raw)

        overall_summary = _compose_overall_summary(argument, evidence, language)
        memory_summary = _compose_memory_summary(argument, evidence, language, phase)

        analysis = TurnAnalysis.compute(
            turn_index=self._turn,
            speaker_id=speaker_id,
            speaker_stance=speaker_stance,
            phase=phase,
            target_id=target_id,
            argument=DimensionResult(**argument),
            source=DimensionResult(**evidence),
            language=DimensionResult(**language),
            overall_summary=overall_summary,
            memory_summary=memory_summary,
        )

        # 4. 분석 결과 → analysis_memory에 저장
        self.memory.add_analysis(
            turn_index=self._turn,
            speaker_stance=speaker_stance,
            analysis=analysis,
        )
        self._turn += 1
        return analysis

    def finalize_phase(self, phase: str, summary: str):
        pass
