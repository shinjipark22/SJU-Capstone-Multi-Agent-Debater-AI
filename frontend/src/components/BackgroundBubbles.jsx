const TEXT_SLOTS = {
  t2: { x: 1228, y: 94, w: 170, h: 44, size: 13 },
  m1: { x: 6, y: 114, w: 348, h: 48, size: 15 },
  m2: { x: 100, y: 425, w: 288, h: 48, size: 14 },
  r1: { x: 1168, y: 398, w: 264, h: 48, size: 13 },
  t1: { x: 475, y: 75, w: 220, h: 40, size: 13 },
  c1: { x: 920, y: 480, w: 300, h: 44, size: 14 },
  b1: { x: 20, y: 580, w: 260, h: 44, size: 13 },
  b2: { x: 1170, y: 720, w: 260, h: 44, size: 13 },
};

const TOPIC_BUBBLE_TEXT = {
  tech: {
    t2: 'AI 윤리',
    m1: '생성형 AI 규제',
    m2: '저작권과 데이터 학습',
    r1: '알고리즘 투명성',
    t1: '딥페이크 규제',
    c1: '개인정보 보호',
    b1: '기술 격차 심화',
    b2: '자동화와 고용',
  },
  economy: {
    t2: '조세 정책',
    m1: '금융투자소득세 도입',
    m2: '시장 활성화 vs 형평성',
    r1: '개인 투자자 부담',
    t1: '부유세 도입',
    c1: '최저임금 인상',
    b1: '소득 불평등',
    b2: '재정 건전성',
  },
  society: {
    t2: '정치 제도',
    m1: '국회의원 불체포특권 폐지',
    m2: '권한 남용 방지',
    r1: '입법 독립성',
    t1: '선거제도 개혁',
    c1: '미디어 공정성',
    b1: '시민 참여',
    b2: '정치 양극화',
  },
  science: {
    t2: '에너지 전환',
    m1: '원자력 발전 확대',
    m2: '안전성과 효율성',
    r1: '탄소중립 해법',
    t1: '재생에너지 전환',
    c1: '기후변화 대응',
    b1: '핵폐기물 처리',
    b2: '에너지 자립도',
  },
};

const BackgroundBubbles = ({ activeTopic }) => {
  const texts = TOPIC_BUBBLE_TEXT[activeTopic] ?? {};

  return (
    <div className="absolute inset-0 overflow-hidden">
      {activeTopic === 'tech' && (
        <img
          src="/technology.png"
          alt=""
          className="absolute inset-0 h-full w-full object-cover"
          style={{ opacity: 0.05 }}
        />
      )}
      {activeTopic === 'economy' && (
        <img
          src="/economy.png"
          alt=""
          className="absolute inset-0 h-full w-full object-cover"
          style={{ opacity: 0.07 }}
        />
      )}
      {activeTopic === 'society' && (
        <img
          src="/justice.png"
          alt=""
          className="absolute inset-0 h-full w-full object-cover"
          style={{ opacity: 0.07 }}
        />
      )}
      {activeTopic === 'science' && (
        <img
          src="/sci.png"
          alt=""
          className="absolute inset-0 h-full w-full object-cover"
          style={{ opacity: 0.07 }}
        />
      )}

      <div className="absolute left-[-16px] top-[1px] h-[999px] w-[1480px]">
        <img
          src="/image.png"
          alt=""
          className="absolute inset-0 h-full w-full max-w-none"
          style={{
            opacity: 0.58,
            filter: 'brightness(0) saturate(0)',
            mixBlendMode: 'multiply',
          }}
        />

        {Object.entries(texts).map(([id, text], index) => {
          const slot = TEXT_SLOTS[id];
          if (!slot) return null;

          return (
            <div
              key={`${activeTopic}-${id}`}
              className="bubble-text-enter absolute flex items-center justify-center px-4 text-center font-semibold tracking-[-0.01em] text-stone-400"
              style={{
                left: slot.x,
                top: slot.y,
                width: slot.w,
                height: slot.h,
                fontSize: slot.size,
                lineHeight: 1.2,
                textShadow: '0 1px 1px rgba(0, 0, 0, 0.04)',
                animationDelay: `${index * 90}ms`,
              }}
            >
              {text}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default BackgroundBubbles;
