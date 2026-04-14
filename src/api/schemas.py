from pydantic import BaseModel


# ── 요청 ─────────────────────────────────────────────
class EvaluateRequest(BaseModel):
    pre_pro: str
    pre_con: str
    post_pro: str
    post_con: str


# ── 응답 ─────────────────────────────────────────────
class MetricScore(BaseModel):
    label: str
    score: int
    reason: str


class PhaseResult(BaseModel):
    evidence_expansion: MetricScore
    knowledge_specificity: MetricScore
    evidence_validity: MetricScore
    reasoning_density: MetricScore
    perspective_diversity: MetricScore
    average_100: float
    overall_summary: str


class SideResult(BaseModel):
    pre: PhaseResult
    post: PhaseResult
    delta_100: float


class EvaluateResponse(BaseModel):
    topic_id: str
    topic: str
    pro_label: str
    con_label: str
    pro: SideResult
    con: SideResult
