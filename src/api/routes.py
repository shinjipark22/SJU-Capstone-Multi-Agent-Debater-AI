from fastapi import APIRouter, HTTPException

from evaluation import analyze_user_before_after
from .schemas import EvaluateRequest, EvaluateResponse, MetricScore, PhaseResult, SideResult

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


def _build_phase_result(phase: dict) -> PhaseResult:
    scores = phase["scores"]
    return PhaseResult(
        evidence_expansion=MetricScore(
            label=scores["evidence_expansion"]["label"],
            score=scores["evidence_expansion"]["score"],
            reason=scores["evidence_expansion"]["reason"],
        ),
        knowledge_specificity=MetricScore(
            label=scores["knowledge_specificity"]["label"],
            score=scores["knowledge_specificity"]["score"],
            reason=scores["knowledge_specificity"]["reason"],
        ),
        evidence_validity=MetricScore(
            label=scores["evidence_validity"]["label"],
            score=scores["evidence_validity"]["score"],
            reason=scores["evidence_validity"]["reason"],
        ),
        reasoning_density=MetricScore(
            label=scores["reasoning_density"]["label"],
            score=scores["reasoning_density"]["score"],
            reason=scores["reasoning_density"]["reason"],
        ),
        perspective_diversity=MetricScore(
            label=scores["perspective_diversity"]["label"],
            score=scores["perspective_diversity"]["score"],
            reason=scores["perspective_diversity"]["reason"],
        ),
        average_100=phase["average_100"],
        overall_summary=phase.get("overall_summary", ""),
    )


@router.post("/", response_model=EvaluateResponse)
async def evaluate(req: EvaluateRequest, topic: str):
    """
    토론 전후 찬/반 근거를 받아 5가지 지표 점수와 평균 점수를 반환합니다.

    - topic: 쿼리 파라미터로 전달 (백엔드 세션에서 주입 가능)
    - pre_pro / pre_con: 토론 전 찬성·반대 근거
    - post_pro / post_con: 토론 후 찬성·반대 근거
    """
    try:
        result = analyze_user_before_after(
            pre_pro=req.pre_pro,
            pre_con=req.pre_con,
            post_pro=req.post_pro,
            post_con=req.post_con,
            topic=topic,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return EvaluateResponse(
        topic=topic,
        pro=SideResult(
            pre=_build_phase_result(result["pro"]["pre"]),
            post=_build_phase_result(result["pro"]["post"]),
            delta_100=result["pro"]["delta_100"],
        ),
        con=SideResult(
            pre=_build_phase_result(result["con"]["pre"]),
            post=_build_phase_result(result["con"]["post"]),
            delta_100=result["con"]["delta_100"],
        ),
    )
