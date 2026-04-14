"""
사용 예시 — Colab에서 실행할 때 이 파일만 열면 됩니다.

다른 모델을 쓰고 싶다면 llm_fn 인자에 어댑터 함수를 넘기세요.
  def my_llm(system_prompt: str, user_prompt: str) -> str:
      return client.responses.create(...)

내 드라이브에 저장된 모델을 쓰려면 evaluation/llm.py의
QWEN_LOCAL_MODEL_PATH를 실제 경로로 바꾸세요.
  예: "/content/drive/MyDrive/models/Qwen2.5-7B-Instruct"
"""

from evaluation import (
    analyze_user_before_after,
    print_evaluation_report,
    save_result_json,
)

TOPIC = "기본소득은 확대되어야 하는가"

PRE_PRO = """
기본소득은 필요하다고 생각한다. 소득이 불안정한 사람이 많아졌고, 최소한의 삶을 보장해줘야 하기 때문이다. 또 소비가 늘어나서 경제에도 도움이 될 수 있다.
""".strip()

PRE_CON = """
기본소득은 반대한다. 재정 부담이 너무 크고, 일할 의욕이 떨어질 수 있다.
""".strip()

POST_PRO = """
기본소득은 확대할 필요가 있다. 첫째, 플랫폼 노동자나 프리랜서처럼 고용이 불안정한 계층이 늘어나 기존 복지 제도만으로는 사각지대를 막기 어렵다. 둘째, 현금 이전은 저소득층의 한계소비성향이 높아 지역 소비를 직접 자극할 수 있다. 셋째, 선별 복지보다 행정비용과 낙인효과를 줄일 수 있다는 장점도 있다. 다만 재정 부담이 크기 때문에 전면 확대보다 청년·실업 취약계층부터 단계적으로 도입하는 방식이 현실적이다.
""".strip()

POST_CON = """
기본소득은 확대에 신중해야 한다. IMF 2023년 보고서에 따르면 전면적 기본소득 도입 시 GDP 대비 10% 이상의 재정이 필요하다. 핀란드 기본소득 실험(2017~2018)에서는 수급자의 취업률이 비수급자와 유의미한 차이가 없었고, 오히려 노동 참여 감소 우려가 확인됐다. 또한 복지 예산을 기본소득으로 전환하면 의료·교육 같은 현물 급여가 축소돼 저소득층에게 더 불리해질 수 있다.
""".strip()


result = analyze_user_before_after(
    pre_pro=PRE_PRO,
    pre_con=PRE_CON,
    post_pro=POST_PRO,
    post_con=POST_CON,
    topic=TOPIC,
)

print_evaluation_report(result)
save_result_json(result)
