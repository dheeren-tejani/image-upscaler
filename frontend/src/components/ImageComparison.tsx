import { useState, useRef, useEffect, useCallback } from 'react';
import { useImageZoomPan } from './useImageZoomPan';

interface ImageComparisonProps {
  originalImage: string;
  enhancedImage: string;
  enhancedImgRef?: React.RefObject<HTMLImageElement>;
}

export const ImageComparison = ({ originalImage, enhancedImage, enhancedImgRef }: ImageComparisonProps) => {
  const [sliderPosition, setSliderPosition] = useState(50);
  const [isDragging, setIsDragging] = useState(false);

  // Use the shared zoom/pan hook
  const {
    containerRef,
    zoom,
    panOffset,
    isPanning,
    handleMouseDown: handlePanMouseDown,
  } = useImageZoomPan();

  const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {

  };

  const handleMouseDown = (event: React.MouseEvent) => {
    // Check if we're clicking on the slider handle
    if ((event.target as HTMLElement).closest('.slider-handle')) {
      setIsDragging(true);
    } else {
      handlePanMouseDown(event);
    }
  };

  const updateSliderPosition = useCallback((percentage: number) => {
    setSliderPosition(Math.max(0, Math.min(100, percentage)));
  }, []);

  const handleMouseMove = useCallback((event: MouseEvent) => {
    // Use a guard clause to exit early if not dragging.
    if (!isDragging || !containerRef.current) return;

    const rect = containerRef.current.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const percentage = (x / rect.width) * 100;

    // Directly update the state without requestAnimationFrame.
    updateSliderPosition(percentage);
  }, [isDragging, containerRef, updateSliderPosition]);

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  useEffect(() => {
    // We only need to add/remove listeners for the entire document when dragging starts/stops.
    if (isDragging) {
      document.addEventListener('mousemove', handleMouseMove);
      document.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      document.removeEventListener('mousemove', handleMouseMove);
      document.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, handleMouseMove]);

  return (
    <div className="flex items-center justify-center w-full h-full relative bg-black rounded-full" style={{ minHeight: '300px', minWidth: '300px' }}>
      {/* Subtle background pattern */}
      <div className="absolute inset-0 opacity-5 bg-[radial-gradient(circle_at_1px_1px,rgba(255,255,255,0.15)_1px,transparent_0)] bg-[length:20px_20px]" />

      {/* Image Container */}
      <div
        ref={containerRef}
        className="relative flex items-center justify-center max-w-[80vw] max-h-[90vh] rounded-3xl overflow-hidden shadow-2xl select-none bg-black backdrop-blur-sm border border-gray-700/20"
        style={{
          transform: `scale(${zoom}) translate(${panOffset.x / zoom}px, ${panOffset.y / zoom}px)`,
          transition: isPanning ? 'none' : 'transform 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
          cursor: zoom > 1.0 ? (isPanning ? 'grabbing' : 'grab') : 'ew-resize',
          boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.05)'
        }}
        onMouseDown={handleMouseDown}
      >
        {/* Enhanced Image (Background) */}
        <img
          ref={enhancedImgRef}
          src={enhancedImage}
          alt="Enhanced"
          className="block object-contain max-w-[80vw] max-h-[90vh] rounded-2xl"
          draggable={false}
        />

        {/* Original Image (Clipped overlay) */}
        <div
          className="absolute inset-0 overflow-hidden rounded-2xl"
          style={{
            clipPath: `inset(0 ${100 - sliderPosition}% 0 0)`,
            transition: isDragging ? 'none' : 'clip-path 0.25s cubic-bezier(0.4, 0, 0.2, 1)'
          }}
        >
          <img
            src={originalImage}
            alt="Original"
            className="block object-contain max-w-[80vw] max-h-[90vh] rounded-2xl"
            draggable={false}
          />
        </div>

        {/* Slider Handle */}
        <div
          className="slider-handle absolute top-0 bottom-0 w-1 bg-gradient-to-b from-white/90 via-white/70 to-white/10 shadow-2xl cursor-ew-resize flex items-center justify-center transition-all duration-200 hover:w-1.5 z-20"
          style={{ left: `${sliderPosition}%`, transform: 'translateX(-50%)', transition: isDragging ? 'none' : 'left 0.25s cubic-bezier(0.4,0,0.2,1)' }}
        >
        </div>

        {/* Labels */}
        <div className="absolute top-6 left-6 bg-gray-900/90 backdrop-blur-xl text-white px-4 py-2 rounded-xl text-sm font-medium font-sans pointer-events-none border border-gray-700/30 shadow-lg">
          Original
        </div>
        <div className="absolute top-6 right-6 bg-gray-900/90 backdrop-blur-xl text-white px-4 py-2 rounded-xl text-sm font-medium font-sans pointer-events-none border border-gray-700/30 shadow-lg">
          Enhanced
        </div>

        {/* Subtle corner highlights */}
        <div className="absolute top-0 left-0 w-32 h-32 bg-gradient-to-br from-white/5 to-transparent rounded-2xl pointer-events-none" />
        <div className="absolute bottom-0 right-0 w-32 h-32 bg-gradient-to-tl from-white/5 to-transparent rounded-2xl pointer-events-none" />
      </div>

      {/* Zoom percentage container */}
      <div className="absolute -bottom-6 right-4 z-20">
        <div className="bg-[#383838] rounded px-3 py-1 shadow flex items-center justify-center w-16 h-8">
          <span className="text-sm font-semibold text-white font-sans">{Math.round(zoom * 100)}%</span>
        </div>
      </div>
    </div>
  );
};