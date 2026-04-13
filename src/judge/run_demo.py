"""
Debatrix 데모 실행 스크립트.
Google Colab에서: python run_demo.py
"""
import json

from .inference import load_model_colab
from .judge import DebatrixJudge, plot_live_debate, _PHASE_KR
from .demo_data import TOPIC, DEBATE_FORMAT, USER_ID, USER_STANCE, TEAMS, DEMO_SPEECHES
from .config import MODEL_ID, PERCENT_BASELINE, PERCENT_STEP


def run_demo():
    load_model_colab()

    judge = DebatrixJudge(TOPIC, DEBATE_FORMAT, USER_ID, USER_STANCE, TEAMS)
    analyses = []
    current_phase = None
    live_history = []

    for speaker_id, stance, phase, content, target_id in DEMO_SPEECHES:
        if current_phase and phase != current_phase:
            print(f"\n{'─'*50}")
            print(f"단계 전환: {_PHASE_KR.get(current_phase)} -> {_PHASE_KR.get(phase)}")
        current_phase = phase

        print(f"\n{'='*55}")
        print(f"T{len(analyses)} | {speaker_id}({'찬성' if stance == 'PRO' else '반대'}) | {_PHASE_KR.get(phase)}")
        print(f"   발언: {content[:70]}...")

        result = judge.judge_turn(
            speaker_id=speaker_id,
            speaker_stance=stance,
            phase=phase,
            speech_content=content,
            target_id=target_id,
        )
        analyses.append(result)

        live = judge.memory.live_debate_snapshot()
        live_history.append({"turn": len(analyses) - 1, **live})

        print(f"   논증: {result.argument.score:4.1f}")
        print(f"   근거: {result.evidence.score:4.1f}")
        print(f"   언어: {result.language.score:4.1f}")
        print(f"   종합: {result.weighted_score:.2f}")
        print(f"   요약: {result.overall_summary}")
        print(f"   현재 비율: 반대 {live['con_percent']:.1f}% | 찬성 {live['pro_percent']:.1f}%")

        plot_live_debate(
            live["pro_percent"],
            live["con_percent"],
            title=f"T{len(analyses)-1} {_PHASE_KR.get(phase)} | 종합 우세 비율",
        )

    print(f"\n{'='*55}")
    print("전체 심사 완료")

    _print_memory_report(judge)
    _save_results(judge, analyses)


def _print_memory_report(judge: DebatrixJudge):
    snap = judge.memory.to_snapshot()

    plot_live_debate(
        snap["live_debate"]["pro_percent"],
        snap["live_debate"]["con_percent"],
        title="최종 종합 우세 비율",
    )

    print("[최종 종합 우세 비율]")
    print(f"  반대 {snap['live_debate']['con_percent']:.1f}% | 찬성 {snap['live_debate']['pro_percent']:.1f}%")

    print("\n[speech_memory 최근 5개]")
    for item in snap["speech_memory"][-5:]:
        print(f"  T{item['turn_index']} {item['speaker_id']} -> {item['summary']}")

    print("\n[analysis_memory 최근 5개]")
    for item in snap["analysis_memory"][-5:]:
        print(
            f"  T{item['turn_index']} {item['speaker_id']} | "
            f"A:{item['argument']['summary']} | "
            f"S:{item['evidence']['summary']} | "
            f"L:{item['language']['summary']}"
        )

    print("\n" + "=" * 60)
    print("SPEECH MEMORY")
    print("=" * 60)
    for item in judge.memory.speech_memory:
        print(f"  T{item['turn_index']} | {item['speaker_id']}({item['speaker_stance']}) | {item['phase']}")
        print(f"      요약: {item['summary']}")
        print()

    print("=" * 60)
    print("ANALYSIS MEMORY")
    print("=" * 60)
    for item in judge.memory.analysis_memory:
        arg = item["argument"]
        evd = item["evidence"]
        lng = item["language"]
        print(
            f"  T{item['turn_index']} | {item['speaker_id']}({item['speaker_stance']}) | "
            f"{item['phase']} | 종합: {item['weighted_score']:.2f}"
        )
        print(f"      논증 {arg['score']:4.1f}  {arg['summary']}")
        print(f"      근거 {evd['score']:4.1f}  {evd['summary']}")
        print(f"      언어 {lng['score']:4.1f}  {lng['summary']}")
        print()

    print("=" * 60)
    print("LIVE DEBATE")
    print("=" * 60)
    snap_live = judge.memory.live_debate_snapshot()
    print(f"  찬성 EMA: {snap_live['pro_ema']:.3f}  |  반대 EMA: {snap_live['con_ema']:.3f}")
    print(f"  찬성: {snap_live['pro_percent']:.1f}%  |  반대: {snap_live['con_percent']:.1f}%")


def _save_results(judge: DebatrixJudge, analyses: list, path: str = "debatrix_result.json"):
    output = {
        "topic": TOPIC,
        "format": DEBATE_FORMAT,
        "teams": TEAMS,
        "model_id": MODEL_ID,
        "quantized_4bit": True,
        "percent_baseline": PERCENT_BASELINE,
        "percent_step": PERCENT_STEP,
        "turn_analyses": [a.model_dump() for a in analyses],
        "live_debate": judge.memory.live_debate_snapshot(),
        "speech_memory": judge.memory.speech_memory,
        "analysis_memory": judge.memory.analysis_memory,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"결과 저장 완료: {path}")
    print(f"   발언 수: {len(analyses)}개 | 파일 크기: {len(json.dumps(output)):,} bytes")


if __name__ == "__main__":
    run_demo()
