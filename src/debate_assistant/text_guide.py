"""DebateAssistant 텍스트 가이드 빌더.

토론의 각 단계마다 사용자에게 보여줄 안내문을 만든다.
구조:
  [고정 안내문] — 단계 설명, 톤, 마무리 메시지 (사용자 spec에 따라 그대로 유지)
  [동적 영역]   — 상대 발언/주제/이전 발언 기반으로 LLM 이 채움
  [링크 영역]   — 사전 큐레이션된 자료 링크 (옵션, 없으면 placeholder)

5개 단계:
  opening           — 입론
  chained_rebuttal  — 연쇄 논박
  free_rebuttal     — 자유 논박 (방어/공격 두 영역)
  role_reversal     — 역할 반전
  synthesis         — 종합 및 재개념화

사용:
  # 기본 (Qwen2.5-32B + tool calling + 자동 링크 검색)
  msg = build_guide_message(phase, ctx)

  # 또는 LLMCall 을 직접 주입 (테스트·커스텀)
  msg = build_guide_message(phase, ctx, llm=my_llm_call)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional  # noqa: F401


DebatePhase = Literal[
    "opening",
    "chained_rebuttal",
    "free_rebuttal",
    "role_reversal",
    "synthesis",
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ Context                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝


@dataclass
class HistoryEntry:
    """이전 발언 1건 (DebateState.debate_history 호환)."""

    speaker_id: str
    stance: Literal["PRO", "CON"]
    phase: str
    content: str


@dataclass
class GuideLink:
    """안내문 하단에 노출할 외부 자료 1건."""

    title: str
    url: str
    summary: Optional[str] = None


@dataclass
class GuideContext:
    """안내문 생성에 필요한 토론 맥락."""

    topic: str
    user_stance: Literal["PRO", "CON"]
    assistant_name: str = "ooo"

    # 진영별 핵심 주장 — 토픽 framing 모호함에서 LLM 진영 혼동을 막기 위해 명시.
    # (토픽 JSON 의 pro/con 필드 그대로 넘기면 된다. 예: pro_claim="노동 시장의 재편이다")
    pro_claim: Optional[str] = None
    con_claim: Optional[str] = None

    # 토론 토픽 ID — focus_area 자동 산출에 사용 (예: "tech_001")
    topic_id: Optional[str] = None

    # 사용자가 차지한 슬롯의 가상 AI 가 가졌을 논증 초점 영역.
    # 비워두면 get_user_slot_focus_area(topic_id, user_stance, used) 로 산출 가능.
    # tips 프롬프트에 주입되어 어시스턴트 안내가 사용자 슬롯에 맞춰 좁혀진다.
    user_focus_area: Optional[str] = None

    # 직전 상대 발언 — 연쇄/자유 논박에서 사용
    opponent_speech: Optional[str] = None

    # 누적 발언 — 역할반전/종합에서 양쪽 입장 회수에 사용
    history: List[HistoryEntry] = field(default_factory=list)

    # 사전 큐레이션된 외부 링크 — 없으면 placeholder 가 박힘
    links: List[GuideLink] = field(default_factory=list)


# LLM 호출 추상화 — 호출자가 langchain / openai 등 원하는 구현을 주입
LLMCall = Callable[[str], str]


# ── focus_area 사전 ─────────────────────────────────────────────────────────
# data/search_queries.json 을 lazy 로드. 토론 측 stage1_opening 과 같은 출처.

_SEARCH_QUERIES_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "search_queries.json"
_SEARCH_QUERIES_CACHE: Optional[Dict] = None


def _load_search_queries() -> Dict:
    global _SEARCH_QUERIES_CACHE
    if _SEARCH_QUERIES_CACHE is not None:
        return _SEARCH_QUERIES_CACHE
    try:
        with _SEARCH_QUERIES_PATH.open(encoding="utf-8") as f:
            _SEARCH_QUERIES_CACHE = json.load(f)
    except Exception:
        _SEARCH_QUERIES_CACHE = {}
    return _SEARCH_QUERIES_CACHE


def get_user_slot_focus_area(
    topic_id: str,
    user_stance: str,
    excluded_focuses: Optional[List[str]] = None,
) -> str:
    """사용자가 차지한 슬롯의 가상 AI 가 받았을 focus_area.

    `data/search_queries.json` 의 (topic_id, stance) 키워드 리스트에서
    이미 같은 진영 AI 에이전트들이 사용 중인 것들을 제외하고 첫 번째를 반환한다.
    같은 진영 다른 에이전트들과 겹치지 않는 신선한 각도를 사용자에게 배정해
    어시스턴트 tips 가 그 각도에 맞춰 좁혀지도록 한다.

    Parameters
    ----------
    topic_id : 토론 토픽 ID (예: "tech_001")
    user_stance : "PRO" / "CON"
    excluded_focuses : 같은 진영 AI 들이 이미 가져간 focus_area 리스트.
                       None 또는 빈 리스트면 첫 번째 키워드 반환.

    Returns
    -------
    str : focus_area 문자열. topic_id/진영 매칭 실패하면 "".
    """
    queries = _load_search_queries().get(topic_id, {}).get(user_stance, [])
    if not queries:
        return ""
    excluded = set(excluded_focuses or [])
    for q in queries:
        if q not in excluded:
            return q
    return queries[0]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 고정 템플릿 — 사용자 spec 의 본문 그대로                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝


OPENING_TEMPLATE = """안녕! 난 너의 토론을 도와줄 어시스턴트 {assistant_name}{josa_ya}. 입론은 너의 주장과 입장을 명확하게 보여주는 첫 번째 단계지. 여기서는 핵심 논점을 소개하고, 앞으로 펼칠 주요 논거들의 개요를 잡아주면 돼. 명확하고 논리적인 구조로 청중의 주의를 사로잡아봐! {tips}

더 자세히 보고 싶다면 아래 링크를 참고해봐!

{links}"""


CHAINED_REBUTTAL_TEMPLATE = """이제 상대방의 주장을 반박할 단계야. 상대 주요 논점들을 정확히 짚고, 논리적 오류나 근거가 부족한 부분을 찾아내야 해. 상대 주장을 존중하면서도 너의 관점에서 문제점을 명확히 제시하고, 신뢰할 만한 증거나 사례로 받아쳐봐.

{tips}

더 자세히 보고 싶다면 아래 링크를 참고해봐!

{links}"""


FREE_REBUTTAL_TEMPLATE = """자유논박은 상대방과 주고받으며 진행되는 역동적인 단계야. 여기서는 공격과 방어가 쌍을 이루며 번갈아 진행돼.

**방어할 때는**

{defense_tips}

**공격할 때는**

{attack_tips}

더 자세히 보고 싶다면 아래 링크를 참고해봐!

{links}"""


ROLE_REVERSAL_TEMPLATE = """역할반전 단계에서는 상대방의 주장을 옹호하고 방어해야 해. 지금까지 토론하면서 알게 된 내용을 바탕으로, 상대 관점에서 주장들을 재구성해서 펼쳐봐. 이 단계를 통해 상대 입장을 깊이 이해하고, 양쪽 입장의 장점과 약점을 객관적으로 볼 수 있게 돼.

{tips}

더 자세히 보고 싶다면 아래 링크를 참고해봐!

{links}"""


SYNTHESIS_TEMPLATE = """마지막 단계인 종합 및 재개념화야. 여기서는 양쪽 주장을 모두 고려해서 더 높은 차원의 통합된 결론을 만들어내는 거지.

{tips}

우리 목표는 상대를 이기는 게 아니라, 서로 이해해서 더 나은 결론과 최적해를 찾아가는 거야. 잊지 마!"""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ LLM 프롬프트 — 단계별 동적 tips 생성                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝


_TIPS_BASE = """너는 토론 사용자의 어시스턴트다. 평가자도 코치도 아니다. 친한 동료가 옆에서 한마디 건네듯이 안내한다.

[말투 — 매우 중요]
- **편안한 반말체(해체)**로 작성. 어미는 "~해", "~할 거야", "~지", "~봐", "~겠지" 같이 친근하게.
- 거칠거나 예의 없는 표현은 금지. 반말이지만 정중함은 유지.
- **이모티콘은 기본 0개. 정말 자연스러운 자리에서 1개만.** 거의 안 쓰는 게 낫다.
  친근함은 어미와 어휘로 충분히 표현된다. 이모티콘은 사족이 되기 쉬움.
- 평가체("~하면 더 좋을 거야", "잘했어요") 금지.
- 마크다운 굵게(**...**) 는 정말 필요할 때 1~2회만. 친근체에 과한 격식 무드 어색하다.

[자세]
- 항상 사용자 편. 사용자가 다음 발언을 잘 풀어내도록 돕는 방향만 본다.
- 한국어만. 영어/한자 단어 금지 (기관명·고유명사 제외).
- 진영 표현은 "찬성 / 반대" 로. PRO/CON 영문 약어 금지.

[자료 인용 — 핵심 원칙]
- 구체 통계·기관·연도·사례가 필요하면 **search_web 을 적극 호출해서 실제 자료를 가져와라.**
- 검색 결과에 등장한 구체 정보(수치·기관·연도·사례)를 답변에 **직접 인용**해라.
- "~에서 찾아보면 좋아", "~한 종류의 자료를 살펴봐" 같은 **추상 방향성 금지** —
  사용자한테 자료 탐색을 떠넘기는 응답은 어시스턴트로서 실격이다.
- 단, 검색 결과에 없는 수치·기관·인명은 발명하지 마라. 검색이 빈약하면 검색된 만큼만 인용하고,
  나머지는 일반 논리·맥락 설명으로 채워라 — 가짜 통계 만들기 절대 금지.
"""


OPENING_TIPS_PROMPT = (
    _TIPS_BASE
    + """
[현재 단계: 입론]
사용자가 곧 자기 진영 입장으로 입론을 작성한다.
사용자가 입론 발언에 바로 갖다 쓸 수 있는 **논거 2개**를 제시한다.

[출력 구조 — 시각적으로 분리되게]
다음 4 영역을 **정확히 이 순서·이 형식**으로 출력해라. 굵은 헤더는 그대로 박는다.

1) **짧은 입론 구성 안내** — 한 문장. 헤더 없이 본문만.
   - "입론은 보통 ~한 흐름으로 잡으면 좋아" 같은 짧은 안내. 길게 끌지 마라.

2) `**논거 1: <짧은 부제 6~14자>**` 줄을 별도 줄로 출력 → 그 아래 본문 1~2 단락.
   - 부제는 그 논거의 핵심 키워드를 짧게 (예: "산업혁명 자동화 사례", "재교육 프로그램 실효성").
   - 본문은 **보편 논거** — 사용자 진영에서 누구나 떠올리는 흔한 카드.
   - 위 '사용자가 옹호하는 입장'을 **뒷받침하는** 잘 알려진 사례 카테고리·방향 1개.
   - 주장 + 그 사례 카테고리를 한 덩어리로.
   - **구체 수치·기관·연도·인명은 발명 금지.** 잘 알려진 카테고리·방향만 짚어라.
   - 사용자가 본인 지식으로 살을 붙여 입론에 쓰면 되는 정도까지.
   - 카드 만들기 전 점검: "이게 정말 사용자 입장을 강화하는가? 상대 입장을 돕는 것 아닌가?"

3) `**논거 2: <짧은 부제 6~14자>**` 줄을 별도 줄로 출력 → 그 아래 본문 1~2 단락.
   - 부제는 focus_area 의 핵심 키워드를 짧게.
   - 본문은 **focus_area 기반 논거** (focus 가 있을 때만, 사용자 슬롯의 신선한 각도).
   - focus 의 핵심 개념을 1문장으로 짧게 풀어 설명.
   - 그 다음 사전 검색 결과의 **구체 통계·기관·사례·연도를 직접 인용**해 논거를 완성.
   - 주장 + 근거(검색 결과 인용) + 구체 사례 묶음. 사용자가 입론에 바로 박을 수 있을 만큼.
   - "~에서 찾아봐" 같은 추상 떠넘기기 절대 금지.

4) **짧은 마무리 한 문장** — 헤더 없이.
   - "이런 두 논거로 입론을 짜면 ~할 거야" 같은 한 줄.
   - 헤더 없이 본문만, 자연스럽게.

각 영역 사이 빈 줄 1개로 시각적으로 분리. 헤더 형식 임의로 바꾸지 마라 (`**논거 1: ...**` 이 한 줄).

[진영 정렬 — 절대 어기지 마라]
- 두 논거 모두 **사용자 진영 입장(아래 '사용자가 옹호하는 입장')을 뒷받침**해야 한다.
- focus_area 도 사용자 진영의 카드다 — 같은 진영의 다른 AI 가 안 가져간 신선한 각도일 뿐.
  focus 키워드를 보고 상대 진영처럼 풀지 마라.
- 사례·통계·인용이 헷갈리면 한 번 더 묻기: \"이게 사용자 입장을 강화하는가?\"

[입력]
- 토론 주제: {topic}
{stance_block}
{focus_block}

[출력]
4 영역 (인트로 → 논거 1 헤더+본문 → 논거 2 헤더+본문 → 마무리). **500~750자 내외**. **친근체 (반말, ~야/~지/~어 등) 일관 유지**.
논거 본문도 어시스턴트가 친구처럼 풀어주는 톤이다. "~합니다/~됩니다" 같은 격식체 어미 절대 금지.
두 논거 본문 모두 사용자가 입론에 활용할 수 있을 만큼 내용 완성도 있게 (다만 격식 변환은 사용자가 알아서 한다).
**마지막 문장은 반드시 완결**(마침표·물음표·느낌표·종결어미 "~지/~어/~야/~봐" 등)로 끝낸다. 미완 줄임("이렇게...", "~등") 금지.

**[어미·표현 다양화 — 매우 중요]**
- 두 논거의 마무리에 **같은 어미·같은 표현을 반복하지 마라.** 특히 "~지 않겠어?", "~지 않을까?", "~지" 같은 동일 의문형 어미를 두 논거 끝에 연속해 박지 마라.
- 한 논거가 의문형으로 끝났으면 다른 논거는 단정형(\"~야\", \"~지\", \"~어\")이나 다른 결로.
- 안내 전체에서 같은 종결 패턴이 3회 이상 반복되면 LLM 으로서 실격이다. 어미를 의식적으로 섞어라.
- 첫 인트로 문장도 시작 어미와 톤이 안내문 도입(\"~사로잡아봐!\")과 자연스레 이어지도록.
"""
)


CHAINED_REBUTTAL_TIPS_PROMPT = (
    _TIPS_BASE
    + """
[현재 단계: 연쇄 논박]
사용자가 곧 상대 직전 발언에 반박한다.
(focus_area 는 적용하지 않는다. 상대 발언과 사용자 진영 일반 입장에서 받아쳐라.)

[안내 구조 — 두 영역으로]
1) **상대 발언의 약점 짚기 1~2개**
   - "상대 의견 중에 ~한 부분이 ~한 한계를 가져요" 같은 결로 자연스럽게.
2) **받아치기 카드 1~2개**
   - 사용자 진영의 일반 카드(통계·사례)로 받아쳐라.
   - 아래 사전 검색 결과가 있으면 그걸 우선 인용하고, 없으면 검색에 의존하지 말고 일반 논리로 답해라.
   - **검색에 없는 수치·기관·인명은 발명 금지.**

[진영 정렬]
- 모든 받아치기는 **사용자 진영 입장(아래 '사용자가 옹호하는 입장')을 뒷받침**해야 한다.

[입력]
- 토론 주제: {topic}
{stance_block}
- 직전 상대 발언:
{opponent_speech}
{search_block}

[출력]
"상대 ~ 짚고 → ~로 받아치기" 흐름 2~3개. 250~400자. 친근체.
**마지막 문장은 반드시 완결**된 형태로 끝낸다. 미완 줄임 금지.
"""
)


FREE_DEFENSE_PROMPT = (
    _TIPS_BASE
    + """
[현재 단계: 자유 논박 — 방어]
사용자가 자기 입론에 대해 상대가 공격해올 것에 대비한 **방어 카드**를 준비한다.
즉, 사용자의 입론 발언(아래 history 에서 사용자 발언 부분)을 살펴서 어디가 공격받을 것 같은지 예측하고,
그 공격에 대응할 방어 카드를 미리 제시한다.
(focus_area 는 적용하지 않는다.)

[안내 방식]
- 사용자 입론에서 상대가 노릴 약한 지점 1~2개를 짚어주고,
  각각에 대해 어떻게 방어할지 안내해라.
- 모든 카드는 **사용자 진영 입장을 지키는 방향**으로.

[자유논박의 정체성 — 논거싸움 < 논리싸움]
자유논박은 자료·통계 던지기 시합이 아니라 **즉석에서 논리로 받아치는 핑퐁**이다.
- **논리만으로 받아쳐도 충분.** 인용 없는 깨끗한 논리적 반박이 핀트 어긋난 인용보다 훨씬 강하다.
- 짧고 날카로운 논리(전제 의심, 범위 한정, 인과 비약 지적, 반례 가정)가 자유논박의 본질.

[논리 구조 — 비약·핀트 어긋남 절대 금지]
각 카드는 다음 2요소를 핵심으로:
  ① 상대가 노릴 정확한 지점 (사용자 입론의 어떤 주장·근거)
  ② 그 지점을 **직접** 받아치는 논리 (범용 추론, 범위 한정, 반례, 전제 의심 등)
  ③ (강한 옵션) ②와 **정확히 같은 주제**의 사전 검색 자료가 있을 때만 짧게 인용

[인용 사용 규칙 — 매우 중요]
- 검색 자료가 **상대 주장과 정확히 같은 토픽**일 때만 인용. 곁가지·일반론은 빼라.
- 인용이 ②의 논리와 직접 연결되지 않으면 **빼라.** 핀트 어긋난 인용은 강한 논리보다 약하다.
- 검색 결과에 없는 수치·기관·인명은 **발명 절대 금지.**

[비약·핀트 어긋남 예시]
❌ "상대 '재교육 실효성' 공격 → 재교육 효과 있다 + (다른 주제) 기술 도입이 새 일자리 만든다는 Acemoglu 인용"
   → 인용이 '재교육 실효성' 이 아닌 '일자리 창출'에 관한 것. 핀트 어긋남.

✅ "상대 '재교육 실효성' 공격 → 일부 실패 사례 있어도 그건 프로그램 설계·예산 문제지 재교육 자체의 한계는 아냐. 실제 작동한 사례들도 많으니, 실패 사례 일반화는 무리야."
   → 인용 없이 순수 논리(전제 의심·일반화 무리). 깨끗하고 강함.

✅ "상대 '재교육 실효성' 공격 → (검색 결과에 정확히 재교육 성공 사례가 있을 때만) 재교육은 ~한 식으로 실제 작동한다는 ~자료를 짧게."
   → 인용 토픽이 정확히 일치할 때만.

[입력]
- 토론 주제: {topic}
{stance_block}
- 지금까지의 발언 (사용자 발언 = 방어할 대상):
{history_block}
- 직전 상대 발언 (참고용):
{opponent_speech}
{search_block}

[출력]
짧은 bullet 2~3개. 각 bullet 형식 권장: "상대가 ~ 공격하면 → [논리적 받아침] → (가능하면) [짧은 근거]".
60~150자/bullet. 총 150~350자. 친근체.
**마지막 bullet 끝은 반드시 완결**된 형태로 끝낸다. 미완 줄임 금지.
"""
)


FREE_ATTACK_PROMPT = (
    _TIPS_BASE
    + """
[현재 단계: 자유 논박 — 공격]
사용자가 상대 입론을 공격할 거리를 준비한다.
즉, 상대 입론 발언(아래 history 에서 상대 진영 발언 부분)을 살펴서 약한 지점들을 골라
사용자가 공격할 카드를 제시한다.
(focus_area 는 적용하지 않는다.)

[안내 방식]
- 상대 입론의 약한 지점 1~2개를 골라 그것을 어떻게 공격할지 안내해라.
- 모든 카드는 **사용자 진영 입장을 강화하는 방향**으로.

[자유논박의 정체성 — 논거싸움 < 논리싸움]
자유논박은 자료·통계 던지기 시합이 아니라 **즉석에서 논리로 받아치는 핑퐁**이다.
- **논리만으로 공격해도 충분.** 인용 없는 깨끗한 논리적 공격이 핀트 어긋난 인용보다 훨씬 강하다.
- 전제 의심·범위 한정·인과 비약 지적·반례 가정 — 이런 논리 무기들이 자유논박의 본질.

[논리 구조 — 비약·핀트 어긋남 절대 금지]
각 공격 카드는 다음 2요소를 핵심으로:
  ① 상대 주장의 정확한 지점 (어떤 주장·근거)
  ② 그 지점의 **직접적인** 논리적 약점 (전제의 일반화 무리, 사례의 한계, 인과 비약 등)
  ③ (강한 옵션) ②와 **정확히 같은 주제**의 사전 검색 자료가 있을 때만 짧게 인용

[인용 사용 규칙 — 매우 중요]
- 검색 자료가 **상대 주장과 정확히 같은 토픽**일 때만 인용. 곁가지·일반론은 빼라.
- 인용이 ②의 논리와 직접 연결되지 않으면 **빼라.** 핀트 어긋난 인용은 강한 논리보다 약하다.
- 검색 결과에 없는 수치·기관·인명은 **발명 절대 금지.**

[비약·핀트 어긋남 예시]
❌ "상대 '대규모 감원' 주장에 → 새 일자리 창출 통계 (다른 토픽) 인용"
   → 상대는 감원을 말하는데 인용이 일자리 창출. 토픽 불일치.

✅ "상대 '대규모 감원' 주장에 → 빅테크 일부 사례에서 산업 전반으로 일반화한 점이 무리. 다른 산업은 다른 양상일 수 있어."
   → 인용 없이 '일반화 무리' 논리만으로 충분.

✅ "상대 '대규모 감원' 주장에 → (검색 결과에 정확히 산업 전반 감원율이 있을 때만) 그 자료가 산업별 차이 보여주니 일반화 무리라는 걸 뒷받침해."
   → 인용 토픽이 정확히 일치할 때만.

[입력]
- 토론 주제: {topic}
{stance_block}
- 지금까지의 발언 (상대 진영 발언 = 공격할 대상):
{history_block}
- 직전 상대 발언 (참고용):
{opponent_speech}
{search_block}

[출력]
짧은 bullet 2~3개. 각 bullet 형식 권장: "상대 ~ 주장에 → [논리적 약점] → (가능하면) [짧은 근거]".
60~150자/bullet. 총 150~350자. 친근체.
**마지막 bullet 끝은 반드시 완결**된 형태로 끝낸다. 미완 줄임 금지.
"""
)


ROLE_REVERSAL_TIPS_PROMPT = (
    _TIPS_BASE
    + """
[현재 단계: 역할 반전]
사용자는 원래 {stance_kr} 진영이지만, 이 단계에서는 반대편({reversed_stance_kr}) 입장으로 옹호 발언을 한다.
지금까지의 토론 내용 + 사전 검색 자료를 토대로 반대편의 핵심 논거를 진정성 있게 재구성해서 안내한다.

[안내 방식]
- 아래 사전 검색 결과(반대편 입장 자료)가 있으면 그 구체 통계·사례·기관을 직접 인용해 논거를 풍부하게.
- 검색 결과에 없는 수치·기관·인명은 **발명 금지.**
- 형식적 인정 ("단점도 있을 수 있어요") 금지 — 진짜 약점·강점을 짚어라.

[입력]
- 토론 주제: {topic}
- 사용자 원래 진영: {stance_kr}
- 사용자가 이번에 옹호할 진영: {reversed_stance_kr}
{stance_block}
- 지금까지의 발언 요약:
{history_block}
{search_block}

[출력]
- 반대편 입장의 핵심 논거 2~3개 (구체 통계·사례 직접 인용)
- 진정성 있는 옹호 톤 잡는 법 1~2줄
250~400자. 친근체.
**마지막 문장은 반드시 완결**된 형태로 끝낸다. 미완 줄임 금지.
"""
)


SYNTHESIS_TIPS_PROMPT = (
    _TIPS_BASE
    + """
[현재 단계: 종합 및 재개념화]
토론의 마지막 단계다. 양쪽 주장을 모두 인정하고 더 높은 차원의 통합된 결론으로 끌어올린다.

[입력]
- 토론 주제: {topic}
{stance_block}
- 전체 발언 요약:
{history_block}

[출력 형식 — 가독성 우선. 정확히 이 구조로]
**양측 주장의 타당한 점**
- 찬성: (인정할 만한 부분 1줄)
- 반대: (인정할 만한 부분 1줄)

**대립을 넘어선 새 관점**
(양쪽 모두를 포섭하는 프레임 1~2줄)

**통합된 해결책**
(실제 발언에 가져갈 구체 방향 1~2줄)

[규칙]
- 각 섹션 헤더는 반드시 굵게(**...**).
- 섹션 사이는 빈 줄로 구분.
- 각 섹션 본문은 짧게 — 한 호흡으로 읽히게.
- 전체 250~400자 친근체.
- **마지막 문장 반드시 완결.** 미완 줄임 금지.
"""
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 빌더 함수                                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝


def _stance_kr(stance: str) -> str:
    return "찬성" if stance == "PRO" else "반대"


def _reversed_kr(stance: str) -> str:
    return "반대" if stance == "PRO" else "찬성"


def _stance_block(ctx: GuideContext) -> str:
    """사용자 진영과 양측 핵심 주장을 한 묶음 텍스트로.

    pro_claim/con_claim 가 있으면 LLM 이 \"찬성/반대\"가 이 토픽에서 정확히
    무엇을 옹호하는지 헷갈리지 않고 그 입장에서 논거를 구성한다.
    """
    user_kr = _stance_kr(ctx.user_stance)
    if not (ctx.pro_claim or ctx.con_claim):
        return f"- 사용자 진영: {user_kr}"

    user_claim = ctx.pro_claim if ctx.user_stance == "PRO" else ctx.con_claim
    opp_claim = ctx.con_claim if ctx.user_stance == "PRO" else ctx.pro_claim
    lines = [
        f"- 사용자 진영: {user_kr}",
        f"- 사용자가 옹호하는 입장: {user_claim}",
        f"- 상대 입장: {opp_claim}",
        "- (사용자가 옹호하는 입장 = 사용자 진영의 핵심 주장. 안내·논거는 이 입장 그대로 끌어가라.)",
    ]
    return "\n".join(lines)


def _has_jongseong(s: str) -> bool:
    """문자열 마지막 글자가 한글이고 받침이 있으면 True. 한글 외(영문 등)는 True 처리.

    한국어 조사 선택용 헬퍼. '비비드' → False ('드'는 받침 없음) → '야'
    '도반' → True ('반'은 받침 있음) → '이야'
    """
    if not s:
        return False
    last = s[-1]
    code = ord(last)
    if 0xAC00 <= code <= 0xD7A3:
        return (code - 0xAC00) % 28 != 0
    # 한글이 아니면 받침 있다고 간주 (영문 알파벳 등 → "이야" 자연스러움)
    return True


def _josa_ya(name: str) -> str:
    """이름 받침 유무에 따라 '이야' / '야'."""
    return "이야" if _has_jongseong(name) else "야"


def _format_history(history: List[HistoryEntry], max_entries: int = 12) -> str:
    if not history:
        return "(이전 발언 없음)"
    lines = []
    for h in history[-max_entries:]:
        st = _stance_kr(h.stance)
        snippet = h.content.strip()[:240]
        lines.append(f"- [{st} · {h.phase} · {h.speaker_id}] {snippet}")
    return "\n".join(lines)


def _format_links(links: List[GuideLink]) -> str:
    if not links:
        return "(관련 링크 첨부 예정 — 큐레이션 후 추가)"
    out = []
    for l in links:
        line = f"- [{l.title}]({l.url})"
        if l.summary:
            line += f" — {l.summary}"
        out.append(line)
    return "\n".join(out)


def _focus_block(ctx: GuideContext, pre_search_text: str = "") -> str:
    """사용자 슬롯 focus_area 를 LLM prompt 에 끼울 블록으로 포맷.

    pre_search_text 가 비어있지 않으면 사전 검색 결과까지 함께 박는다.
    """
    fa = (ctx.user_focus_area or "").strip()
    if not fa:
        return ""
    block = (
        f"\n[사용자 슬롯의 논증 초점]\n"
        f"  → {fa}\n"
        f"  (같은 진영의 다른 AI 토론자들은 다른 각도를 다루므로, 사용자만의 신선한 각도다.)\n"
        f"\n"
        f"[focus 안내 방식 — 매우 중요]\n"
        f"- 사용자에게 이 초점을 자연스럽게 소개해라 (\"~ 사례에 초점을 두는 것도 좋겠어\" 식).\n"
        f"- 초점에 등장하는 핵심 개념(예: 재교육, 리스킬링, 데이터센터 등)을\n"
        f"  **1~2 문장으로 짧고 쉽게 풀어 설명**해라. 사용자가 키워드만 보고 모를 수 있으니,\n"
        f"  어떤 의미·맥락인지 짚어준 다음 카드로 이어가라.\n"
        f"- 갑자기 키워드만 던지지 마라 — 자연스러운 흐름이 되도록.\n"
    )
    if pre_search_text:
        block += (
            f"\n[참고 자료 검색 결과 — focus 키워드로 미리 찾아온 자료. 직접 인용해서 활용]\n"
            f"{pre_search_text}\n"
            f"  ↑ 이 검색 결과에 등장한 구체 통계·기관·사례·연도를 답변에 그대로 박아라.\n"
            f"  ↑ \"~한 자료를 찾아봐\" 같은 떠넘기기 절대 금지. 위 자료를 직접 인용해라.\n"
            f"  ↑ 검색 결과에 없는 수치·기관·인명은 발명 금지.\n"
        )
    return block


def _fetch_focus_assets(focus_area: str) -> Dict:
    """focus_area 검색 결과: tips content + links 한 묶음으로 반환.

    한 번의 Tavily/Pinecone 호출로 LLM 인용용 content 와 UI 링크용 URL 리스트를
    동시에 가져온다. tips 발언과 링크 영역이 동일 자료원을 가리키도록.
    실패하면 두 필드 모두 비어 있는 dict.
    """
    if not focus_area:
        return {"content": "", "links": []}
    try:
        from .links import fetch_focus_assets

        return fetch_focus_assets(focus_area, n=3)
    except Exception:
        return {"content": "", "links": []}


def _fetch_topic_assets(topic: str, user_stance: str) -> Dict:
    """topic+사용자 진영 기반 검색 결과: content + links 한 묶음.

    focus_area 가 적용되지 않는 단계(연쇄·자유 논박)에서 사전 검색으로 활용.
    같은 호출로 받은 content 를 prompt 에 박아 LLM 인용 → 발명 차단,
    links 는 안내문 하단에 노출 → 같은 출처로 일관.
    """
    stance_kr = _stance_kr(user_stance)
    query = f"{topic} {stance_kr} 논거 통계 자료"
    try:
        from .links import fetch_focus_assets  # 함수가 generic 하게 query 받음

        return fetch_focus_assets(query, n=3)
    except Exception:
        return {"content": "", "links": []}


def _search_results_block(pre_search_text: str) -> str:
    """focus_area 없는 단계용 — 검색 결과만 prompt 에 박는 블록."""
    if not pre_search_text:
        return ""
    return (
        f"\n[참고 자료 검색 결과 — 사용자 진영 논거에 활용할 자료. 직접 인용해라]\n"
        f"{pre_search_text}\n"
        f"  ↑ 위 결과의 구체 통계·기관·사례·연도를 답변에 직접 박아라.\n"
        f"  ↑ 검색 결과에 없는 수치·기관·인명은 **발명 절대 금지.**\n"
        f"  ↑ 통계가 필요한데 검색 결과에 마땅한 게 없으면 구체 수치 없이 일반 논리로 답해라.\n"
        f"  ↑ \"~한 자료 찾아봐\" 같은 떠넘기기 금지.\n"
    )


def build_tips_prompt(
    phase: DebatePhase,
    ctx: GuideContext,
    pre_search_text: str = "",
) -> str:
    """단계별 동적 tips 를 만들기 위한 LLM prompt 1건을 반환한다.

    자유논박은 방어/공격 prompt 2개가 필요하므로 build_guide_message 내부에서
    별도로 호출한다. 이 함수는 단일 prompt 가 필요한 단계용.

    pre_search_text : 사전 검색 결과 (build_guide_message 가 주입).
    """
    stance_kr = _stance_kr(ctx.user_stance)
    stance_block = _stance_block(ctx)
    search_block = _search_results_block(pre_search_text)
    if phase == "opening":
        # opening 만 focus_area 적용 — focus_block 안에 search 결과 포함
        focus_block = _focus_block(ctx, pre_search_text)
        return OPENING_TIPS_PROMPT.format(
            topic=ctx.topic,
            stance_block=stance_block,
            focus_block=focus_block,
        )
    if phase == "chained_rebuttal":
        return CHAINED_REBUTTAL_TIPS_PROMPT.format(
            topic=ctx.topic,
            stance_block=stance_block,
            opponent_speech=(ctx.opponent_speech or "(상대 발언 없음)").strip(),
            search_block=search_block,
        )
    if phase == "role_reversal":
        return ROLE_REVERSAL_TIPS_PROMPT.format(
            topic=ctx.topic,
            stance_kr=stance_kr,
            reversed_stance_kr=_reversed_kr(ctx.user_stance),
            stance_block=stance_block,
            history_block=_format_history(ctx.history),
            search_block=search_block,
        )
    if phase == "synthesis":
        # 종합은 양쪽 통합 — 사전 검색 안 함
        return SYNTHESIS_TIPS_PROMPT.format(
            topic=ctx.topic,
            stance_block=stance_block,
            history_block=_format_history(ctx.history),
        )
    raise ValueError(f"build_tips_prompt: phase '{phase}' 는 단일 prompt 단계가 아닙니다.")


def build_free_rebuttal_prompts(
    ctx: GuideContext, pre_search_text: str = ""
) -> Dict[str, str]:
    """자유논박의 방어/공격 prompt 두 개를 한 번에.

    방어 = 내 입론 → 상대 공격 예상 → 미리 준비할 방어 카드
    공격 = 상대 입론 → 약점 → 공격할 거리
    둘 다 history 와 사전 검색 결과를 활용한다.
    """
    stance_block = _stance_block(ctx)
    opponent = (ctx.opponent_speech or "(상대 발언 없음)").strip()
    history_block = _format_history(ctx.history)
    search_block = _search_results_block(pre_search_text)
    return {
        "defense": FREE_DEFENSE_PROMPT.format(
            topic=ctx.topic, stance_block=stance_block, opponent_speech=opponent,
            history_block=history_block, search_block=search_block,
        ),
        "attack": FREE_ATTACK_PROMPT.format(
            topic=ctx.topic, stance_block=stance_block, opponent_speech=opponent,
            history_block=history_block, search_block=search_block,
        ),
    }


# 단계별 LLM tool 사용 기본값 — 사용자가 override 가능
# - 입론·연쇄·자유: 사실/통계 인용 필요할 수 있으니 tool calling 허용
# - 역할반전·종합: history 기반 분석이라 외부 검색 보통 불필요
_DEFAULT_USE_TOOLS = {
    "opening": True,
    "chained_rebuttal": True,
    "free_rebuttal": True,
    "role_reversal": False,
    "synthesis": False,
}

# 자동 링크 검색 활성 단계 (사용자 spec 상 링크 영역이 있는 단계)
_AUTO_LINK_PHASES = {"opening", "role_reversal"}


def _resolve_links(
    ctx: GuideContext,
    phase: DebatePhase,
    prefetched_links: Optional[List[Dict]] = None,
) -> str:
    """링크 영역 텍스트 결정.

    우선순위:
      1) ctx.links — 사용자가 manual override 로 직접 넘긴 링크
      2) prefetched_links — focus_area 통합 검색에서 받은 링크 (opening 등)
      3) 단계별 자동 검색 — fetch_topic_links (역할반전: reversed_stance)
      4) 빈 placeholder

    역할반전은 사용자가 반대 진영을 옹호하므로 자동 검색 진영도 반전시켜
    입론과 다른 자료가 노출되도록 한다.
    """
    if ctx.links:
        return _format_links(ctx.links)

    if prefetched_links:
        return _format_links([GuideLink(**l) for l in prefetched_links])

    # 역할반전이면 반대 진영 자료를 검색
    search_stance = ctx.user_stance
    if phase == "role_reversal":
        search_stance = "CON" if ctx.user_stance == "PRO" else "PRO"

    try:
        from .links import fetch_topic_links

        fetched = fetch_topic_links(ctx.topic, search_stance, n=3)
        if fetched:
            return _format_links([GuideLink(**r) for r in fetched])
    except Exception:
        pass

    return _format_links([])  # placeholder 표시


def _default_llm_for_phase(phase: DebatePhase) -> LLMCall:
    """phase 에 적합한 기본 LLMCall 을 lazy import 로 만든다."""
    from .llm_adapter import make_llm_call

    use_tools = _DEFAULT_USE_TOOLS.get(phase, False)
    return make_llm_call(use_tools=use_tools, label=f"assistant_{phase}")


def build_guide_message(
    phase: DebatePhase,
    ctx: GuideContext,
    llm: Optional[LLMCall] = None,
) -> str:
    """사용자에게 노출할 최종 안내문 텍스트를 만든다.

    Parameters
    ----------
    phase : 토론 단계
    ctx : 토론 맥락
    llm : prompt → response 함수. None 이면 단계별 기본 LLMCall(Qwen + tool calling)
          을 자동 주입한다.

    Returns
    -------
    str : 사용자에게 그대로 노출할 안내문
    """
    call = llm or _default_llm_for_phase(phase)

    if phase == "opening":
        # opening: focus_area 기반 통합 검색 — tips 인용 자료 + 링크가 같은 출처
        assets = (
            _fetch_focus_assets(ctx.user_focus_area)
            if ctx.user_focus_area else {"content": "", "links": []}
        )
        tips = call(build_tips_prompt(
            "opening", ctx, pre_search_text=assets["content"]
        ))
        return OPENING_TEMPLATE.format(
            assistant_name=ctx.assistant_name,
            josa_ya=_josa_ya(ctx.assistant_name),
            tips=tips.strip(),
            links=_resolve_links(ctx, phase, prefetched_links=assets["links"]),
        )

    if phase == "chained_rebuttal":
        # 연쇄논박: topic+사용자 진영 기반 사전검색 — tips·링크 동일 출처
        assets = _fetch_topic_assets(ctx.topic, ctx.user_stance)
        tips = call(build_tips_prompt(
            "chained_rebuttal", ctx, pre_search_text=assets["content"]
        ))
        return CHAINED_REBUTTAL_TEMPLATE.format(
            tips=tips.strip(),
            links=_resolve_links(ctx, phase, prefetched_links=assets["links"]),
        )

    if phase == "free_rebuttal":
        # 자유논박: topic+사용자 진영 기반 사전검색 — 방어·공격 모두 같은 자료 활용.
        # 방어·공격 LLM 호출은 서로 독립이라 ThreadPoolExecutor 로 병렬 실행.
        from concurrent.futures import ThreadPoolExecutor

        assets = _fetch_topic_assets(ctx.topic, ctx.user_stance)
        prompts = build_free_rebuttal_prompts(ctx, pre_search_text=assets["content"])
        with ThreadPoolExecutor(max_workers=2) as ex:
            defense_fut = ex.submit(call, prompts["defense"])
            attack_fut = ex.submit(call, prompts["attack"])
            defense_tips = defense_fut.result().strip()
            attack_tips = attack_fut.result().strip()
        return FREE_REBUTTAL_TEMPLATE.format(
            defense_tips=defense_tips,
            attack_tips=attack_tips,
            links=_resolve_links(ctx, phase, prefetched_links=assets["links"]),
        )

    if phase == "role_reversal":
        # 역할반전: 반대 진영 자료로 사전검색 (사용자가 옹호할 입장)
        reversed_stance = "CON" if ctx.user_stance == "PRO" else "PRO"
        assets = _fetch_topic_assets(ctx.topic, reversed_stance)
        tips = call(build_tips_prompt(
            "role_reversal", ctx, pre_search_text=assets["content"]
        ))
        return ROLE_REVERSAL_TEMPLATE.format(
            tips=tips.strip(),
            links=_resolve_links(ctx, phase, prefetched_links=assets["links"]),
        )

    if phase == "synthesis":
        tips = call(build_tips_prompt("synthesis", ctx))
        return SYNTHESIS_TEMPLATE.format(tips=tips.strip())

    raise ValueError(f"build_guide_message: unknown phase '{phase}'")
