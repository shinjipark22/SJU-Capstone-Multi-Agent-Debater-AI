import React from 'react';
import { Check } from 'lucide-react';

const SubTopicView = ({ activeData, selectedSubTopics, onToggle, visible }) => {
  return (
    <div className={`absolute w-full px-4 flex flex-col items-center justify-center h-full transition-all duration-700 ease-[cubic-bezier(0.34,1.56,0.64,1)]
      ${!visible ? 'opacity-0 -translate-x-32 pointer-events-none' : 'opacity-100 translate-x-0 delay-100'}
    `}>
      {activeData && (
        <div className="flex flex-col items-center transform -translate-y-16 w-full">
          <activeData.icon
            size={72}
            strokeWidth={1.5}
            className="mb-6"
            style={{ color: activeData.accent }}
          />
          <h1 className="text-5xl md:text-6xl font-extrabold text-stone-800 mb-4 tracking-tight text-center">
            {activeData.title}
          </h1>
          <p className="text-xl text-stone-400 mb-16 text-center max-w-xl">
            원하는 세부 토론 논제를 자유롭게 선택해주세요.
          </p>

          <div className="flex w-full max-w-3xl flex-col gap-3">
            {activeData.subTopics.length === 0 && (
              <p className="text-center text-stone-400">논제를 불러오는 중입니다...</p>
            )}
            {activeData.subTopics.map((sub) => {
              const isSelected = selectedSubTopics.some((t) => t.id === sub.id);
              return (
                <button
                  key={sub.id}
                  onClick={() => onToggle(sub)}
                  className={`flex items-start gap-3 rounded-3xl px-6 py-5 text-left transition-all duration-300 ${
                    isSelected
                      ? 'bg-stone-900 text-white shadow-xl'
                      : 'bg-white text-stone-600 border border-stone-200 shadow-sm hover:border-stone-300 hover:shadow-md'
                  }`}
                >
                  {isSelected
                    ? <Check size={20} className="mt-1 shrink-0 text-emerald-400" />
                    : <div className="mt-1 h-5 w-5 shrink-0 rounded-full border-2 border-stone-300" />
                  }
                  <span>
                    <span className="block text-lg font-semibold leading-snug">{sub.title}</span>
                    {sub.description_short && (
                      <span className={`mt-1 block text-sm ${isSelected ? 'text-stone-300' : 'text-stone-400'}`}>
                        {sub.description_short}
                      </span>
                    )}
                  </span>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
};

export default SubTopicView;
