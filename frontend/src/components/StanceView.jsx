import React from 'react';
import { ThumbsUp, ThumbsDown } from 'lucide-react';

const StanceView = ({ userStance, setUserStance, userIntensity, setUserIntensity, visible }) => {
  const intensityLabels = ['약하게', '조금', '보통', '강하게', '매우 강하게'];
  const color = userStance === 'pro' ? '#3b82f6' : '#f87171';

  return (
    <div className={`absolute inset-0 flex flex-col items-center justify-start pt-12 px-8 transition-all duration-700 ease-[cubic-bezier(0.34,1.56,0.64,1)]
      ${!visible ? 'opacity-0 translate-x-32 pointer-events-none' : 'opacity-100 translate-x-0 delay-100'}
    `}>
      <div className="text-center mb-8">
        <h2 className="text-4xl md:text-5xl font-extrabold text-stone-800 tracking-tight">나의 입장</h2>
        <p className="text-stone-400 text-lg mt-2">입장을 선택한 뒤 주장 강도를 설정하세요</p>
      </div>

      <div className="w-full flex items-center justify-center gap-0">

        {/* 찬반 버튼 대각선 배치 */}
        <div className="relative shrink-0" style={{ width: '390px', height: '390px' }}>
          <div className="absolute top-0 left-0">
            <button
              onClick={() => setUserStance('pro')}
              className={`w-44 h-44 md:w-52 md:h-52 rounded-full flex flex-col items-center justify-center gap-3 transition-all duration-500
                ${userStance === 'pro'
                  ? 'bg-white text-stone-900 scale-110 shadow-2xl ring-4 ring-stone-200'
                  : userStance === 'con'
                  ? 'bg-stone-100/60 text-stone-300 border border-stone-200 scale-90'
                  : 'bg-white text-stone-700 border border-stone-200 shadow-sm hover:shadow-lg hover:scale-105'
                }`}
            >
              <ThumbsUp size={44} className={userStance === 'pro' ? 'text-blue-500' : 'text-stone-400'} />
              <span className="text-2xl font-extrabold">찬성</span>
            </button>
          </div>

          <div className="absolute bottom-0 right-0">
            <button
              onClick={() => setUserStance('con')}
              className={`w-44 h-44 md:w-52 md:h-52 rounded-full flex flex-col items-center justify-center gap-3 transition-all duration-500
                ${userStance === 'con'
                  ? 'bg-white text-stone-900 scale-110 shadow-2xl ring-4 ring-stone-200'
                  : userStance === 'pro'
                  ? 'bg-stone-100/60 text-stone-300 border border-stone-200 scale-90'
                  : 'bg-white text-stone-700 border border-stone-200 shadow-sm hover:shadow-lg hover:scale-105'
                }`}
            >
              <ThumbsDown size={44} className={userStance === 'con' ? 'text-rose-500' : 'text-stone-400'} />
              <span className="text-2xl font-extrabold">반대</span>
            </button>
          </div>
        </div>

        {/* 구분선 */}
        <div className="shrink-0 mx-16 self-stretch flex items-center">
          <div className="w-px h-64 bg-stone-200 rounded-full" />
        </div>

        {/* 크기가 다른 원 선택 */}
        <div className={`shrink-0 flex flex-col items-center gap-2 transition-all duration-500
          ${userStance ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-8 pointer-events-none'}`}>
          <p className="text-stone-700 font-bold text-xl mb-1">주장 강도</p>

          {/* 원형 배치 */}
          <div className="relative" style={{ width: 360, height: 360 }}>
            {[1, 2, 3, 4, 5].map((level, i) => {
              const angle = (-90 - i * 72) * (Math.PI / 180);
              const orbitalR = 116;
              const cx = 180 + orbitalR * Math.cos(angle);
              const cy = 180 + orbitalR * Math.sin(angle);
              const size = 58 + (level - 1) * 13;
              const selected = userIntensity === level;
              return (
                <button
                  key={level}
                  onClick={() => setUserIntensity(level)}
                  style={{
                    position: 'absolute',
                    width: size,
                    height: size,
                    left: cx - size / 2,
                    top: cy - size / 2,
                  }}
                  className={`rounded-full flex items-center justify-center font-extrabold transition-all duration-300
                    ${selected
                      ? userStance === 'pro'
                        ? 'bg-blue-500 text-white shadow-lg ring-4 ring-blue-100 scale-110'
                        : 'bg-rose-500 text-white shadow-lg ring-4 ring-rose-100 scale-110'
                      : 'bg-white text-stone-400 border border-stone-200 shadow-sm hover:shadow-md hover:text-stone-600'
                    }`}
                >
                  <span style={{ fontSize: Math.max(10, size * 0.3) }}>+{level}</span>
                </button>
              );
            })}

            {/* 중앙: 선택된 강도 표시 */}
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none" style={{ width: 360, height: 360 }}>
              <span className="text-stone-800 text-3xl font-extrabold">+{userIntensity}</span>
              <span style={{ color }} className="text-sm font-semibold mt-1 transition-all duration-300">
                {intensityLabels[userIntensity - 1]}
              </span>
            </div>
          </div>
        </div>

      </div>
    </div>
  );
};

export default StanceView;
