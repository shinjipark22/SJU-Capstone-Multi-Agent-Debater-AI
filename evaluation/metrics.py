METRICS = [
    {
        "key": "evidence_expansion",
        "label": "근거 확장성",
        "alias": "depth",
        "description": "답변에 새롭고 서로 다른 논거가 얼마나 충분히 포함되어 있는지 평가합니다. 비슷한 근거의 반복은 증가로 보지 않습니다.",
    },
    {
        "key": "knowledge_specificity",
        "label": "지식의 구체성",
        "alias": "specificity",
        "description": "수치, 고유명사, 사례, 제도, 데이터처럼 구체적인 정보가 얼마나 포함되어 있는지 평가합니다.",
    },
    {
        "key": "evidence_validity",
        "label": "근거 타당성",
        "alias": "support",
        "description": "제시한 근거가 최종 주장이나 결론을 실제로 얼마나 잘 뒷받침하는지 평가합니다.",
    },
    {
        "key": "reasoning_density",
        "label": "논리 추론 밀도",
        "alias": "inference",
        "description": "근거에서 결론으로 이어지는 논리 연결이 자연스럽고 탄탄한지, 논리적 비약이 적은지 평가합니다.",
    },
    {
        "key": "perspective_diversity",
        "label": "관점 다각성",
        "alias": "diversity",
        "description": "윤리, 경제, 사회, 정책, 기술 등 서로 다른 관점에서 입체적으로 접근하는지 평가합니다.",
    },
]

METRIC_LABELS = [metric["label"] for metric in METRICS]
