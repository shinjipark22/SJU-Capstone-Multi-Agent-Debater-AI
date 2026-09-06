import { useRef } from 'react';
import { Bot } from 'lucide-react';

const TRACK_HEIGHT = 160;
const INDICATOR_HEIGHT = 40;
const MAX_INDICATOR_BOTTOM = TRACK_HEIGHT - INDICATOR_HEIGHT;

const AiSliderItem = ({ side, index, val, onChange }) => {
  const isPro = side === 'pro';
  const trackRef = useRef(null);
  const fillColor = isPro ? '#77A8F2' : '#FF737D';
  const indicatorBottom =
    ((val - 1) / 4) * MAX_INDICATOR_BOTTOM;
  const fillHeight = indicatorBottom + INDICATOR_HEIGHT;
  const fillPercent = (fillHeight / TRACK_HEIGHT) * 100;

  const handleClick = (e) => {
    const rect = trackRef.current.getBoundingClientRect();
    const offsetY = Math.min(Math.max(e.clientY - rect.top, 0), rect.height);
    const stepHeight = rect.height / 5;
    const level = Math.min(5, Math.max(1, 5 - Math.floor(offsetY / stepHeight)));
    onChange(side, index, level);
  };

  const handleDrag = (e) => {
    if (e.buttons !== 1) return;
    handleClick(e);
  };

  return (
    <div className="flex min-h-[260px] flex-col items-center justify-center gap-2 px-3 py-2 select-none">
      <span className="font-semibold text-[15px] text-stone-800">
        AI #{index + 1}
      </span>

      <span className="text-[12px] font-semibold text-stone-700">+5</span>

      {/* One UI 볼륨바 */}
      <div
        ref={trackRef}
        onClick={handleClick}
        onMouseMove={handleDrag}
        className="relative w-10 rounded-3xl cursor-pointer overflow-visible"
        style={{ height: TRACK_HEIGHT, background: 'rgba(0,0,0,0.07)' }}
      >
        {/* 채워지는 부분 */}
        <div
          className="absolute bottom-0 left-0 right-0 z-10 rounded-3xl transition-all duration-200"
          style={{
            height: `${fillPercent}%`,
            background: fillColor,
            boxShadow: `0 10px 18px ${fillColor}45`,
          }}
        />
        {/* 상단 인디케이터 */}
        <div
          className="absolute left-0 right-0 z-20 rounded-3xl bg-white shadow-[0_8px_16px_rgba(0,0,0,0.18)] transition-all duration-200"
          style={{ height: INDICATOR_HEIGHT, bottom: indicatorBottom }}
        />
      </div>

      <span className="mt-1 text-[12px] font-semibold leading-none text-stone-700">+1</span>

      <span className={`mt-1 rounded-full px-3 py-1 text-[13px] font-bold text-white shadow-[0_6px_10px_rgba(0,0,0,0.18)] ${isPro ? 'bg-[#77A8F2]' : 'bg-[#FF737D]'}`}>
        +{val}
      </span>
    </div>
  );
};

export default AiSliderItem;
