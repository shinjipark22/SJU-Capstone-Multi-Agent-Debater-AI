import { useEffect, useMemo, useState } from 'react';
import { X, ChevronLeft } from 'lucide-react';
import { TOPICS } from './data/topics';
import { getTopics } from './lib/api';
import TopicGrid from './components/TopicGrid';
import BackgroundBubbles from './components/BackgroundBubbles';
import SubTopicView from './components/SubTopicView';
import StanceView from './components/StanceView';
import ParamsView from './components/ParamsView';
import FloatingActionBar from './components/FloatingActionBar';
import DebatePage from './components/debate/DebatePage';

const App = () => {
  const [activeTopic, setActiveTopic] = useState(null);
  const [selectedSubTopics, setSelectedSubTopics] = useState([]);
  const [stage, setStage] = useState(0); // 0: 세부주제, 1: 찬반, 2: 참여설정
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

  const activeData = topicCards.find(t => t.id === activeTopic);
  const selectedTopic = selectedSubTopics[0] ?? null;

  const preDebateBackground = userStance === 'pro'
    ? 'linear-gradient(to bottom right, rgba(147,197,253,0.38), rgba(219,234,254,0.22), rgba(245,245,244,0.08))'
    : userStance === 'con'
    ? 'linear-gradient(to bottom right, rgba(252,165,165,0.38), rgba(254,226,226,0.22), rgba(245,245,244,0.08))'
    : activeData
    ? `linear-gradient(to bottom right, ${activeData.accent}22, rgba(245,245,244,0.1), rgba(245,245,244,0.02))`
    : 'transparent';

  const resetParams = () => {
    setStage(0);
    setUserStance(null);
    setAgentCount(1);
    setMode('debate');
    setInitRequest(null);
  };

  const handleTopicClick = (id) => {
    setActiveTopic(id);
    setSelectedSubTopics([]);
    resetParams();
  };

  const handleClose = () => {
    setActiveTopic(null);
    setSelectedSubTopics([]);
    resetParams();
  };

  // 한 세션은 논제 하나만 다루므로 단일 선택.
  const toggleSubTopic = (sub) => {
    setSelectedSubTopics(prev => (prev[0]?.id === sub.id ? [] : [sub]));
  };

  // 실험 조건 통제를 위해 강경도는 UI 에서 받지 않고 사용자·AI 모두 균형형(3)으로 고정한다.
  // AI 수는 1:1 → 1명, 2:2 → 3명 (백엔드 DEBATE_FORMAT_AI_COUNT).
  const FIXED_INTENSITY = 3;

  const handleEnter = () => {
    if (!selectedTopic) return;

    setInitRequest({
      topic: selectedTopic.id,
      user_stance: userStance === 'pro' ? 'PRO' : 'CON',
      user_intensity: FIXED_INTENSITY,
      agent_intensities: Array(agentCount * 2 - 1).fill(FIXED_INTENSITY),
      debate_format: `${agentCount}:${agentCount}`,
      mode,
      nickname: nickname.trim() || null,
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

      {/* [1] 메인 화면 */}
      <div className={`absolute inset-0 px-4 transition-all duration-700 z-10 overflow-auto
        ${activeTopic ? 'opacity-0 scale-90 pointer-events-none' : 'opacity-100 scale-100'}`}
      >
        <div className="mx-auto flex min-h-screen w-full max-w-[1440px] flex-col items-center pt-8 md:pt-0">
          <nav className="hidden md:flex absolute top-[34px] left-1/2 z-20 -translate-x-1/2 items-center gap-[74px] text-[20px] font-bold leading-[23px] text-black">
            <button className="transition-opacity hover:opacity-70">서비스 소개</button>
            <button className="transition-opacity hover:opacity-70">팀 소개</button>
            <button className="transition-opacity hover:opacity-70">업데이트 소식</button>
          </nav>

          <div className="mt-20 text-center md:mt-[96px]">
            <h1 className="text-[38px] font-extrabold leading-tight tracking-[-0.03em] text-[#38332E] md:text-[48px] md:leading-[55px]">
              어떤 주제에 대해 토론할까요?
            </h1>
            <p className="mt-4 text-[16px] font-medium text-[#393634] md:text-[18px] md:leading-[21px]">
              원하는 카테고리를 선택하여 세부 논제를 확인해보세요.
            </p>
            {topicsError && (
              <p className="mx-auto mt-4 max-w-lg rounded-2xl bg-rose-50 px-4 py-3 text-sm text-rose-600">
                서버에서 논제를 불러오지 못했습니다 ({topicsError}). VITE_API_BASE_URL 설정과 서버 상태를 확인해주세요.
              </p>
            )}
          </div>

          <div className="mt-6 w-full min-w-[1200px] xl:-mt-1">
            <TopicGrid onTopicClick={handleTopicClick} />
          </div>
        </div>
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
              className={`p-4 bg-white/70 hover:bg-white text-stone-600 rounded-full backdrop-blur-md border border-stone-200 shadow-sm transition-all duration-300 hover:scale-110 ${stage > 0 ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-4 pointer-events-none'}`}
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
          <SubTopicView
            activeData={activeData}
            selectedSubTopics={selectedSubTopics}
            onToggle={toggleSubTopic}
            visible={stage === 0}
          />
          <StanceView
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
        onNext={() => setStage(1)}
        onNextStage={() => setStage(2)}
        onEnter={handleEnter}
      />}
    </div>
  );
};

export default App;
