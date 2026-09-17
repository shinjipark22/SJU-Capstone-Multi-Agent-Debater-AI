import React from 'react';
import { ThumbsUp, ThumbsDown } from 'lucide-react';

// 논제를 먼저 보여주고, 찬성/반대가 각각 어떤 입장인지 버튼에 함께 적는다.
const StanceView = ({ topic, userStance, setUserStance, visible }) => {
  const sideButton = (side, Icon, label, claim, activeColor) => {
    const selected = userStance === side;
    const dimmed = userStance && !selected;
    return (
      <button
        onClick={() => setUserStance(side)}
        className={`flex w-[300px] flex-col items-center gap-3 rounded-[32px] px-6 py-8 transition-all duration-500
          ${selected
            ? 'bg-white text-stone-900 scale-105 shadow-2xl ring-4 ring-stone-200'
            : dimmed
            ? 'bg-stone-100/60 text-stone-300 border border-stone-200 scale-95'
            : 'bg-white text-stone-700 border border-stone-200 shadow-sm hover:shadow-lg hover:scale-[1.03]'
          }`}
      >
        <Icon size={40} className={selected ? activeColor : 'text-stone-400'} />
        <span className="text-2xl font-extrabold">{label}</span>
        <span className={`text-center text-[15px] leading-snug ${dimmed ? 'text-stone-300' : 'text-stone-500'}`}>
          {claim}
        </span>
      </button>
    );
  };

  return (
    <div className={`absolute inset-0 flex flex-col items-center justify-start px-8 pt-8 transition-all duration-700 ease-[cubic-bezier(0.34,1.56,0.64,1)]
      ${!visible ? 'opacity-0 translate-x-32 pointer-events-none' : 'opacity-100 translate-x-0 delay-100'}
    `}>
      {/* 논제 */}
      <div className="mb-10 w-full max-w-3xl text-center">
        <span className="inline-block rounded-full bg-stone-100 px-4 py-1 text-sm font-semibold text-stone-500">
          오늘의 논제
        </span>
        <h2 className="mt-4 text-[26px] font-extrabold leading-snug tracking-tight text-stone-800 md:text-[32px]">
          {topic?.title ?? '논제를 불러오는 중...'}
        </h2>
        {topic?.description_short && (
          <p className="mt-4 text-[15px] leading-relaxed text-stone-500 md:text-base">
            {topic.description_short}
          </p>
        )}
      </div>

      <p className="mb-6 text-lg font-medium text-stone-500">어느 입장에서 토론에 참여할지 선택하세요</p>

      <div className="flex flex-wrap items-stretch justify-center gap-6">
        {sideButton('pro', ThumbsUp, '찬성', topic?.pro ?? '', 'text-blue-500')}
        {sideButton('con', ThumbsDown, '반대', topic?.con ?? '', 'text-rose-500')}
      </div>
    </div>
  );
};

export default StanceView;
