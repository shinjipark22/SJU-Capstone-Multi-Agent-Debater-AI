import React from 'react';
import { TOPICS } from '../data/topics';

const DESKTOP_LAYOUT = {
  tech: { size: 280, left: 312, top: 60, icon: 52 },
  society: { size: 284, left: 688, top: 60, icon: 56 },
  economy: { size: 286, left: 442, top: 328, icon: 54 },
  science: { size: 282, left: 834, top: 328, icon: 56 },
};

const TopicGrid = ({ onTopicClick }) => {
  return (
    <div className="w-full">
      <div className="relative h-[660px]">
        {TOPICS.map((topic) => {
          const Icon = topic.icon;
          const layout = DESKTOP_LAYOUT[topic.id];
          return (
            <div
              key={topic.id}
              onClick={() => onTopicClick(topic.id)}
              className={`absolute overflow-hidden flex cursor-pointer flex-col items-center justify-center rounded-full text-white shadow-[0_5px_16px_rgba(0,0,0,0.22)] transition-transform duration-300 hover:scale-[1.03] hover:shadow-[0_8px_24px_rgba(0,0,0,0.24)] ${topic.color}`}
              style={{
                width: layout.size,
                height: layout.size,
                left: layout.left,
                top: layout.top,
              }}
            >
              <img
                src={topic.cardImage}
                alt=""
                className="absolute inset-0 h-full w-full object-cover"
                style={{ opacity: 0.2, mixBlendMode: 'screen' }}
              />
              <div className="absolute inset-0 bg-black/8" />
              <div className="relative z-10 flex flex-col items-center justify-center">
                <Icon size={layout.icon} className="mb-4 opacity-90" strokeWidth={1.5} />
                <h2 className="mb-1 text-[22px] font-bold tracking-tight">{topic.title}</h2>
                <p className="text-center text-[14px] font-medium text-white/80">{topic.desc}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default TopicGrid;
