import { Cpu, TrendingUp, Landmark, Leaf } from 'lucide-react';

export const TOPICS = [
  {
    id: 'tech',
    title: '기술/AI',
    desc: '인공지능',
    color: 'bg-[#4F46E5]',
    cardImage: '/technology.png',
    lightBg: 'bg-gradient-to-br from-indigo-100 via-indigo-50 to-[#F5F5F4]',
    accent: '#4F46E5',
    icon: Cpu,
    subTopics: ['생성형 AI 규제']
  },
  {
    id: 'economy',
    title: '경제/산업',
    desc: '경제 정책',
    color: 'bg-[#059669]',
    cardImage: '/economy.png',
    lightBg: 'bg-gradient-to-br from-emerald-100 via-emerald-50 to-[#F5F5F4]',
    accent: '#059669',
    icon: TrendingUp,
    subTopics: ['금융투자소득세 도입']
  },
  {
    id: 'society',
    title: '정치/사회',
    desc: '정책',
    color: 'bg-[#D97706]',
    cardImage: '/justice.png',
    lightBg: 'bg-gradient-to-br from-amber-100 via-amber-50 to-[#F5F5F4]',
    accent: '#D97706',
    icon: Landmark,
    subTopics: ['국회의원 불체포특권 폐지']
  },
  {
    id: 'science',
    title: '과학/환경',
    desc: '기후 변화',
    color: 'bg-[#0891B2]',
    cardImage: '/sci.png',
    lightBg: 'bg-gradient-to-br from-sky-100 via-sky-50 to-[#F5F5F4]',
    accent: '#0891B2',
    icon: Leaf,
    subTopics: ['원자력 발전 확대지']
  }
];
