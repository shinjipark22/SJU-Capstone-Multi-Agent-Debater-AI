// 발언자 표시 이름. 백엔드(persona_factory / stage1 opening)와 같은 규칙:
//   - 에이전트 번호는 상대 진영 먼저, 같은 진영 나중 (1:1 → agent_1=상대, 2:2 → agent_1·2=상대, agent_3=같은 편)
//   - 진영별로 순서를 세어 "찬성1", "반대2" 처럼 부른다. 참가자는 "나".
export function makeSpeakerLabeler(initRequest) {
  const perSide = parseInt(String(initRequest?.debate_format ?? '1:1').split(':')[0], 10) || 1;
  const userStance = initRequest?.user_stance ?? 'PRO';
  const opposing = userStance === 'PRO' ? 'CON' : 'PRO';

  return (speakerId) => {
    if (!speakerId || speakerId === 'user') return '나';
    const num = parseInt(String(speakerId).replace('agent_', ''), 10);
    if (Number.isNaN(num)) return speakerId;
    const stance = num <= perSide ? opposing : userStance;
    const ordinal = num <= perSide ? num : num - perSide;
    return `${stance === 'PRO' ? '찬성' : '반대'}${ordinal}`;
  };
}
