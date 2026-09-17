import React from 'react';

const ParamsView = ({
  activeData,
  selectedSubTopics,
  agentCount,
  setAgentCount,
  nickname,
  setNickname,
  mode,
  setMode,
  visible,
}) => {
  return (
    <div className={`absolute w-full h-full px-4 flex flex-col items-center transition-all duration-700 ease-[cubic-bezier(0.34,1.56,0.64,1)] hide-scrollbar overflow-y-auto pb-80
      ${!visible ? 'opacity-0 translate-x-32 pointer-events-none' : 'opacity-100 translate-x-0 delay-100'}
    `}>
      {activeData && (
        <div className="w-full max-w-4xl flex flex-col items-center pt-8 md:pt-12">
          <div className="mb-8 text-center">
            <span className="inline-block px-4 py-1 rounded-full bg-stone-100 text-stone-500 text-sm font-semibold mb-4">
              {selectedSubTopics[0]?.title}
            </span>
            <h2 className="text-4xl md:text-5xl font-extrabold text-stone-800 tracking-tight">
              참여 설정
            </h2>
          </div>

          {/* 진행 방식 — 설문지 종류(토론/구성적 논쟁)와 토론 단계 수를 함께 결정한다 */}
          <div className="mb-10 flex w-full flex-col items-center">
            <h3 className="mb-3 text-lg font-medium text-stone-500">진행 방식</h3>
            <div className="flex gap-3">
              {[
                { value: 'debate', label: '토론', desc: '입론 · 연쇄논박 · 자유논박' },
                { value: 'constructive', label: '구성적 논쟁', desc: '역할반전 · 종합까지' },
              ].map((option) => (
                <button
                  key={option.value}
                  onClick={() => setMode(option.value)}
                  className={`rounded-3xl px-7 py-4 text-center transition-all duration-300 ${
                    mode === option.value
                      ? 'bg-stone-900 text-white shadow-xl'
                      : 'border border-stone-200 bg-white text-stone-600 shadow-sm hover:border-stone-300'
                  }`}
                >
                  <span className="block text-lg font-bold">{option.label}</span>
                  <span className={`text-xs ${mode === option.value ? 'text-stone-300' : 'text-stone-400'}`}>
                    {option.desc}
                  </span>
                </button>
              ))}
            </div>
          </div>

          {/* 참여자 닉네임 — 수집 데이터(nickname 컬럼)에 기록된다 */}
          <div className="mb-10 flex w-full max-w-sm flex-col items-center">
            <h3 className="mb-3 text-lg font-medium text-stone-500">닉네임 (선택)</h3>
            <input
              value={nickname}
              onChange={(e) => setNickname(e.target.value)}
              placeholder="토론 기록에 남길 이름"
              maxLength={20}
              className="w-full rounded-full border border-stone-200 bg-white px-6 py-3 text-center text-base outline-none focus:border-stone-400"
            />
          </div>

          {/* 참여 규모 */}
          <div className="mb-10 w-full flex flex-col items-center">
            <h3 className="text-lg font-medium text-stone-500 mb-5 text-center">참여 규모 (각 진영 동수)</h3>
            <div className="flex justify-center gap-6">
              {[1, 2].map(num => (
                <button
                  key={`agent-${num}`}
                  onClick={() => setAgentCount(num)}
                  className={`w-28 h-28 rounded-full flex flex-col items-center justify-center transition-all duration-300 ${
                    agentCount === num
                      ? 'bg-stone-900 text-white scale-105 shadow-2xl'
                      : 'bg-white text-stone-700 border border-stone-200 shadow-sm hover:shadow-md hover:scale-105'
                  }`}
                >
                  <span className="text-xl font-bold">{num}v{num}</span>
                  <span className="text-xs font-medium text-stone-400">총 {num * 2}명</span>
                </button>
              ))}
            </div>
          </div>

          {/* AI 성향은 실험 조건 통제를 위해 균형형(3)으로 고정 */}
          <p className="mb-32 text-sm text-stone-400">AI 에이전트 성향은 균형형으로 동일하게 설정됩니다.</p>
        </div>
      )}
    </div>
  );
};

export default ParamsView;
