import React from 'react';

const FloatingActionBar = ({ selectedSubTopics, stage, userStance, onNext, onNextStage, onEnter }) => {
  const showBar = selectedSubTopics.length > 0;

  return (
    <div className={`fixed bottom-10 left-1/2 -translate-x-1/2 z-40 transition-all duration-500 ease-[cubic-bezier(0.34,1.56,0.64,1)] ${
      showBar ? 'translate-y-0 opacity-100 scale-100' : 'translate-y-24 opacity-0 scale-90'
    }`}>
      {stage === 0 && (
        <button
          onClick={onNext}
          className="flex items-center gap-4 px-10 py-5 bg-stone-900 text-white rounded-full font-bold text-xl hover:bg-black hover:scale-105 active:scale-95 transition-all shadow-2xl"
        >
          <span className="flex items-center justify-center w-8 h-8 bg-emerald-500 text-white rounded-full text-base">
            {selectedSubTopics.length}
          </span>
          다음 단계로
        </button>
      )}

      {stage === 1 && (
        <button
          onClick={onNextStage}
          disabled={!userStance}
          className={`flex items-center gap-4 px-10 py-5 rounded-full font-bold text-xl transition-all shadow-2xl ${
            userStance
              ? 'bg-stone-900 text-white hover:bg-black hover:scale-105 active:scale-95 cursor-pointer'
              : 'bg-stone-200 text-stone-400 cursor-not-allowed'
          }`}
        >
          {userStance ? '참여자 수 선택하기' : '입장을 선택해주세요'}
        </button>
      )}

      {stage === 2 && (
        <button
          onClick={onEnter}
          className="flex items-center gap-4 px-12 py-5 bg-stone-900 text-white rounded-full font-bold text-xl hover:bg-black hover:scale-105 active:scale-95 transition-all shadow-2xl cursor-pointer"
        >
          토론방 입장하기
        </button>
      )}
    </div>
  );
};

export default FloatingActionBar;
