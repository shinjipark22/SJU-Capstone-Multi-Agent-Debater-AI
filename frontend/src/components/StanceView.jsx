import React from 'react';
import { ThumbsUp, ThumbsDown } from 'lucide-react';

const StanceView = ({ userStance, setUserStance, visible }) => {
  return (
    <div className={`absolute inset-0 flex flex-col items-center justify-start pt-12 px-8 transition-all duration-700 ease-[cubic-bezier(0.34,1.56,0.64,1)]
      ${!visible ? 'opacity-0 translate-x-32 pointer-events-none' : 'opacity-100 translate-x-0 delay-100'}
    `}>
      <div className="text-center mb-12">
        <h2 className="text-4xl md:text-5xl font-extrabold text-stone-800 tracking-tight">나의 입장</h2>
        <p className="text-stone-400 text-lg mt-2">어느 쪽에서 토론에 참여할지 선택하세요</p>
      </div>

      <div className="flex items-center justify-center gap-10 md:gap-16">
        <button
          onClick={() => setUserStance('pro')}
          className={`w-44 h-44 md:w-56 md:h-56 rounded-full flex flex-col items-center justify-center gap-3 transition-all duration-500
            ${userStance === 'pro'
              ? 'bg-white text-stone-900 scale-110 shadow-2xl ring-4 ring-stone-200'
              : userStance === 'con'
              ? 'bg-stone-100/60 text-stone-300 border border-stone-200 scale-90'
              : 'bg-white text-stone-700 border border-stone-200 shadow-sm hover:shadow-lg hover:scale-105'
            }`}
        >
          <ThumbsUp size={48} className={userStance === 'pro' ? 'text-blue-500' : 'text-stone-400'} />
          <span className="text-2xl font-extrabold">찬성</span>
        </button>

        <button
          onClick={() => setUserStance('con')}
          className={`w-44 h-44 md:w-56 md:h-56 rounded-full flex flex-col items-center justify-center gap-3 transition-all duration-500
            ${userStance === 'con'
              ? 'bg-white text-stone-900 scale-110 shadow-2xl ring-4 ring-stone-200'
              : userStance === 'pro'
              ? 'bg-stone-100/60 text-stone-300 border border-stone-200 scale-90'
              : 'bg-white text-stone-700 border border-stone-200 shadow-sm hover:shadow-lg hover:scale-105'
            }`}
        >
          <ThumbsDown size={48} className={userStance === 'con' ? 'text-rose-500' : 'text-stone-400'} />
          <span className="text-2xl font-extrabold">반대</span>
        </button>
      </div>
    </div>
  );
};

export default StanceView;
