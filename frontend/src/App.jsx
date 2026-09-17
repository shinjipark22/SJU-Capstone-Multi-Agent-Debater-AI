import { useEffect, useMemo, useState } from 'react';
import { X, ChevronLeft } from 'lucide-react';
import { TOPICS } from './data/topics';
import { getTopics } from './lib/api';
import BackgroundBubbles from './components/BackgroundBubbles';
import StanceView from './components/StanceView';
import ParamsView from './components/ParamsView';
import FloatingActionBar from './components/FloatingActionBar';
import DebatePage from './components/debate/DebatePage';

// 실험은 이 논제 하나로만 진행한다. 주제 선택 화면을 건너뛰고 바로 찬반 선택으로 들어간다.
const FIXED_TOPIC_ID = 'tech_003';

const App = () => {
  const [stage, setStage] = useState(1); // 1: 찬반, 2: 참여설정, 3: 토론 (0: 세부주제 선택은 사용 안 함)
  const [userStance, setUserStance] = useState(null);
  const [agentCount, setAgentCount] = useState(1);
  const [nickname, setNickname] = useState('');
  const [mode, setMode] = useState('debate');
  const [backendTopics, setBackendTopics] = useState({});
  const [topicsError, setTopicsError] = useState(null);
  const [initRequest, setInitRequest] = useState(null);

  // 세부 논제는 백엔드 /topics 가 원본 (topic id 를 그대로 /debate/init 에 넘겨야 함).
  // 카테고리 키('기술/AI' 등)가 카드 title 과 동일해 title 로 매칭한다.
  useEffect(() => {
    const controller = new AbortController();
    getTopics(controller.signal)
      .then(res => setBackendTopics(res.categories ?? {}))
      .catch(e => {
        if (e.name !== 'AbortError') setTopicsError(String(e.detail ?? e.message ?? e));
      });
    return () => controller.abort();
  }, []);

  const topicCards = useMemo(
    () => TOPICS.map(card => ({ ...card, subTopics: backendTopics[card.title] ?? [] })),
    [backendTopics],
  );

  // 논제는 고정이므로 상태가 아니라 논제 목록에서 그대로 도출한다 (목록이 오기 전엔 null).
  const activeData = useMemo(
    () => topicCards.find(c => c.subTopics.some(t => t.id === FIXED_TOPIC_ID)) ?? null,
    [topicCards],
  );
  const activeTopic = activeData?.id ?? null;
  const selectedTopic = activeData?.subTopics.find(t => t.id === FIXED_TOPIC_ID) ?? null;
  const selectedSubTopics = selectedTopic ? [selectedTopic] : [];

  const preDebateBackground = userStance === 'pro'
    ? 'linear-gradient(to bottom right, rgba(147,197,253,0.38), rgba(219,234,254,0.22), rgba(245,245,244,0.08))'
    : userStance === 'con'
    ? 'linear-gradient(to bottom right, rgba(252,165,165,0.38), rgba(254,226,226,0.22), rgba(245,245,244,0.08))'
    : activeData
    ? `linear-gradient(to bottom right, ${activeData.accent}22, rgba(245,245,244,0.1), rgba(245,245,244,0.02))`
    : 'transparent';

  const resetParams = () => {
    setStage(1);
    setUserStance(null);
    setAgentCount(1);
    setMode('debate');
    setInitRequest(null);
  };

  // 닫기·새 토론: 논제는 고정이므로 찬반 선택으로만 되돌린다.
  const handleClose = () => {
    resetParams();
  };

  // 실험 조건 통제를 위해 강경도는 UI 에서 받지 않고 사용자·AI 모두 균형형(3)으로 고정한다.
  // AI 수는 1:1 → 1명, 2:2 → 3명 (백엔드 DEBATE_FORMAT_AI_COUNT).
  const FIXED_INTENSITY = 3;

  const handleEnter = () => {
    if (!selectedTopic || !nickname.trim()) return;

    setInitRequest({
      topic: selectedTopic.id,
      user_stance: userStance === 'pro' ? 'PRO' : 'CON',
      user_intensity: FIXED_INTENSITY,
      agent_intensities: Array(agentCount * 2 - 1).fill(FIXED_INTENSITY),
      debate_format: `${agentCount}:${agentCount}`,
      mode,
      nickname: nickname.trim(),
    });
    setStage(3);
  };

  return (
    <div className="relative isolate min-h-screen w-full overflow-hidden bg-[#F5F5F4] text-gray-800" style={{ fontFamily: 'var(--ui-font)' }}>
      <div className="pointer-events-none fixed inset-0 z-0">
        <BackgroundBubbles activeTopic={activeTopic} />
        <div
          className="absolute inset-x-0 top-0 h-[84px] md:h-[100px]"
          style={{
            background: 'linear-gradient(to bottom, rgba(0,0,0,0.42), rgba(0,0,0,0.12), transparent)',
          }}
        />
        <div className="absolute inset-0 bg-[#F5F5F4]/0" />
      </div>
      <style>{`
        .hide-scrollbar::-webkit-scrollbar { display: none; }
        .hide-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>

      {/* [1] 논제 로드 대기 — 고정 논제라 카테고리 선택 화면은 쓰지 않는다 */}
      <div className={`absolute inset-0 z-10 flex items-center justify-center px-4 transition-all duration-700
        ${activeTopic ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}
      >
        {topicsError ? (
          <p className="max-w-lg rounded-2xl bg-rose-50 px-4 py-3 text-sm text-rose-600">
            서버에서 논제를 불러오지 못했습니다 ({topicsError}). 서버 상태를 확인해주세요.
          </p>
        ) : (
          <p className="text-lg font-medium text-stone-400">토론을 준비하는 중...</p>
        )}
      </div>

      {/* [3] 풀스크린 오버레이 */}
      <div
        className={`fixed inset-0 z-30 flex flex-col transition-all duration-700 delay-300
        ${activeTopic ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
        style={stage < 3 ? { background: preDebateBackground } : undefined}
      >
        {/* 토론 페이지가 아닐 때만 네비 버튼 표시 */}
        {stage < 3 && (
          <div className="absolute top-8 left-8 right-8 md:top-12 md:left-12 md:right-12 flex justify-between z-50">
            <button
              onClick={() => setStage(prev => prev - 1)}
              className={`p-4 bg-white/70 hover:bg-white text-stone-600 rounded-full backdrop-blur-md border border-stone-200 shadow-sm transition-all duration-300 hover:scale-110 ${stage > 1 ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-4 pointer-events-none'}`}
            >
              <ChevronLeft size={28} />
            </button>
            <button
              onClick={handleClose}
              className="p-4 bg-white/70 hover:bg-white text-stone-600 rounded-full backdrop-blur-md border border-stone-200 shadow-sm transition-all duration-300 hover:scale-110 ml-auto"
            >
              <X size={28} />
            </button>
          </div>
        )}

        <div className={`relative flex-1 flex justify-center w-full ${stage === 2 ? 'overflow-visible' : 'overflow-hidden'} ${stage < 3 ? 'mt-24 md:mt-10' : ''}`}>
          <StanceView
            topic={selectedTopic}
            userStance={userStance}
            setUserStance={setUserStance}
            visible={stage === 1}
          />
          <ParamsView
            activeData={activeData}
            selectedSubTopics={selectedSubTopics}
            agentCount={agentCount}
            setAgentCount={setAgentCount}
            nickname={nickname}
            setNickname={setNickname}
            mode={mode}
            setMode={setMode}
            visible={stage === 2}
          />
          {initRequest && selectedTopic && (
            <DebatePage
              topic={selectedTopic}
              initRequest={initRequest}
              visible={stage === 3}
              onRestart={handleClose}
            />
          )}
        </div>
      </div>

      {/* [4] 하단 플로팅 액션 바 - 토론 중에는 숨김 */}
      {stage < 3 && <FloatingActionBar
        selectedSubTopics={selectedSubTopics}
        stage={stage}
        userStance={userStance}
        canEnter={Boolean(nickname.trim())}
        onNext={() => setStage(1)}
        onNextStage={() => setStage(2)}
        onEnter={handleEnter}
      />}
    </div>
  );
};

export default App;
