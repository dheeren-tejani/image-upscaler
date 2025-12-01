import { useState, useEffect } from 'react';
import { Sparkles, Download, ChevronDown } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group';

interface ControlPanelProps {
  originalImagePreviewURL: string | null;
  selectedUpscaleFactor: number;
  onUpscaleFactorChange: (factor: number) => void;
  selectedMode: 'Normal' | 'Crazy';
  onModeChange: (mode: 'Normal' | 'Crazy') => void;
  onEnhance: () => void;
  onDownload: () => void;
  isUpscaling: boolean;
  hasUpscaledImage: boolean;
  errorMessage: string | null;
  inputImageInfo?: { width: number; height: number; size: number } | null;
  enhancedImageInfo?: { width: number; height: number; size: number } | null;
  isVideoMode?: boolean;
  onVideoModeChange?: (isVideo: boolean) => void;
}

function formatBytes(bytes: number) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KBs', 'MBs', 'GBs'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

export const ControlPanel = ({
  originalImagePreviewURL,
  selectedUpscaleFactor,
  onUpscaleFactorChange,
  selectedMode,
  onModeChange,
  onEnhance,
  onDownload,
  isUpscaling,
  hasUpscaledImage,
  errorMessage,
  inputImageInfo,
  enhancedImageInfo,
  isVideoMode = false,
  onVideoModeChange
}: ControlPanelProps) => {
  const [settingsExpanded, setSettingsExpanded] = useState(true);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });

  // Update image dimensions when image changes
  useEffect(() => {
    if (originalImagePreviewURL) {
      const img = new Image();
      img.onload = () => {
        setImageDimensions({ width: img.width, height: img.height });
      };
      img.src = originalImagePreviewURL;
    } else {
      setImageDimensions({ width: 0, height: 0 });
    }
  }, [originalImagePreviewURL]);

  const getOutputDimensions = () => {
    if (!originalImagePreviewURL || imageDimensions.width === 0) return { width: 0, height: 0 };
    return {
      width: imageDimensions.width * selectedUpscaleFactor,
      height: imageDimensions.height * selectedUpscaleFactor
    };
  };

  const { width, height } = getOutputDimensions();

  return (
    <div className="w-[260px] bg-[#121212] border-l border-[#121212] flex flex-col">
      {/* Preview Section */}
      <div className="p-4 border-b border-[#121212]">
        {originalImagePreviewURL ? (
          <div className="w-full h-56 rounded-2xl border border-[#121212] overflow-hidden shadow-2xl bg-gradient-to-br from-gray-900/30 to-gray-800/30 backdrop-blur-sm">
            <img
              src={originalImagePreviewURL}
              alt="Preview"
              className="w-full h-full object-cover"
            />
          </div>
        ) : (
          <div className="w-full h-56 rounded-xl border border-[#121212]  bg-gradient-to-br from-[#393939] to-[#171717] backdrop-blur-sm flex items-center justify-center shadow-inner">
            <span className="text-gray-400 text-sm font-medium">No image loaded</span>
          </div>
        )}
      </div>

      {/* Image/Video Mode Toggle */}
      {onVideoModeChange && (
        <div className="p-4 border-b border-gray-800/50">
          <div className="relative w-full bg-gray-800/60 rounded-xl p-1">
            <div
              className="absolute top-1 h-[calc(100%-8px)] rounded-lg bg-white shadow-lg transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)] z-0"
              style={{
                width: 'calc(50% - 4px)',
                left: !isVideoMode ? '4px' : 'calc(50% + 2px)',
              }}
            />
            <div className="relative z-10 flex w-full">
              <button
                onClick={() => onVideoModeChange(false)}
                className={`flex-1 rounded-lg text-base font-semibold transition-all duration-300 py-2 ${!isVideoMode ? 'text-black' : 'text-gray-300'}`}
              >
                Image
              </button>
              <button
                onClick={() => onVideoModeChange(true)}
                className={`flex-1 rounded-lg text-base font-semibold transition-all duration-300 py-2 ${isVideoMode ? 'text-black' : 'text-gray-300'}`}
              >
                Video
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Upscale Factor Slider */}
      <div className="p-4 border-b border-gray-800/50">
        <div className="flex flex-col items-center space-y-3">
          <div className="relative w-full bg-gray-800/60 rounded-xl p-1">
            {/* Sliding button */}
            <div
              className="absolute top-1 h-[calc(100%-8px)] rounded-lg bg-white shadow-lg transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)] z-0"
              style={{
                width: isVideoMode ? 'calc(50% - 4px)' : 'calc(33.33% - 4px)',
                left: isVideoMode
                  ? (selectedUpscaleFactor === 2 ? '4px' : 'calc(50% + 2px)')
                  : (selectedUpscaleFactor === 2 ? '4px' : selectedUpscaleFactor === 4 ? 'calc(33.33% + 2px)' : 'calc(66.67% + 2px)'),
              }}
            />
            <div className="relative z-10 flex w-full">
              <button
                onClick={() => onUpscaleFactorChange(2)}
                className={`flex-1 rounded-lg text-base font-semibold transition-all duration-300 py-2 ${selectedUpscaleFactor === 2 ? 'text-black' : 'text-gray-300'}`}
              >
                2x
              </button>
              <button
                onClick={() => onUpscaleFactorChange(4)}
                className={`flex-1 rounded-lg text-base font-semibold transition-all duration-300 py-2 ${selectedUpscaleFactor === 4 ? 'text-black' : 'text-gray-300'}`}
              >
                4x
              </button>
              {!isVideoMode && (
                <button
                  onClick={() => onUpscaleFactorChange(8)}
                  className={`flex-1 rounded-lg text-base font-semibold transition-all duration-300 py-2 ${selectedUpscaleFactor === 8 ? 'text-black' : 'text-gray-300'}`}
                >
                  8x
                </button>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Mode Selection Slider (only shows when 2x is selected) */}
      <div className={`overflow-hidden transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)] ${selectedUpscaleFactor === 2 ? 'max-h-24 opacity-100' : 'max-h-0 opacity-0'}`}>
        <div className="p-4 border-b border-[#121212]">
          <div className="flex flex-col items-center space-y-3">
            <div className="relative w-full bg-[#383838] rounded-xl p-1">
              {/* Sliding button */}
              <div
                className="absolute top-1 h-[calc(100%-8px)] rounded-lg bg-white shadow-lg transition-all duration-500 ease-[cubic-bezier(0.4,0,0.2,1)] z-0"
                style={{
                  width: 'calc(50% - 4px)',
                  left: selectedMode === 'Normal' ? '4px' : 'calc(50% + 2px)',
                }}
              />
              <div className="relative z-10 flex w-full">
                <button
                  onClick={() => onModeChange('Normal')}
                  className={`flex-1 rounded-lg text-base font-semibold transition-all duration-300 py-2 ${selectedMode === 'Normal' ? 'text-black' : 'text-gray-300'}`}
                >
                  Normal
                </button>
                <button
                  onClick={() => onModeChange('Crazy')}
                  className={`flex-1 rounded-lg text-base font-semibold transition-all duration-300 py-2 ${selectedMode === 'Crazy' ? 'text-black' : 'text-gray-300'}`}
                >
                  Crazy
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Info Section */}
      <div className="flex-1 overflow-y-auto scrollbar-hide" style={{
        scrollbarWidth: 'none',
        msOverflowStyle: 'none',
      }}>
        <style jsx>{`
          .scrollbar-hide::-webkit-scrollbar {
            display: none; /* Safari and Chrome */
          }
        `}</style>
        <div className="p-4">
          <Button
            variant='ghost'
            onClick={() => setSettingsExpanded(!settingsExpanded)}
            className="w-full justify-between text-gray-300 mb-4 rounded-xl font-medium bg-[#121212] hover:text-gray-300 hover:bg-[#121212]"
          >
            Info
            <ChevronDown className={`w-4 h-4 transition-transform duration-200 ${settingsExpanded ? '' : 'rotate-90'}`} />
          </Button>

          <div className={`overflow-hidden transition-all duration-300 ease-out ${settingsExpanded ? 'max-h-96 opacity-100' : 'max-h-0 opacity-0'}`}>
            <div className="space-y-6 font-sans" style={{ fontFamily: 'Inter, sans-serif' }}>
              {/* Input Image Info */}
              <div className="bg-[#393939] p-4 rounded-2xl border border-gray-800/50 backdrop-blur-sm">
                <label className="text-sm text-gray-400 mb-3 block font-medium font-sans">Input Image</label>
                <div className="flex flex-col space-y-1 text-gray-300 text-sm font-sans">
                  <span className="leading-relaxed">Resolution : {inputImageInfo ? `${inputImageInfo.width} x ${inputImageInfo.height}` : '--'}</span>
                  <span className="leading-relaxed">File Size : {inputImageInfo ? formatBytes(inputImageInfo.size) : '--'}</span>
                </div>
              </div>
              {/* Enhanced Image Info */}
              <div className="bg-[#393939] p-4 rounded-2xl border border-gray-800/50 backdrop-blur-sm">
                <label className="text-sm text-gray-400 mb-3 block font-medium font-sans">Enhanced Image ({selectedUpscaleFactor}x)</label>
                <div className="flex flex-col space-y-1 text-gray-300 text-sm font-sans">
                  <span className="leading-relaxed">Resolution : {enhancedImageInfo ? `${enhancedImageInfo.width} x ${enhancedImageInfo.height}` : '--'}</span>
                  <span className="leading-relaxed">File Size : {enhancedImageInfo ? formatBytes(enhancedImageInfo.size) : '--'}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="p-4 border-t border-[#121212] space-y-3">
        {errorMessage && (
          <div className="text-red-400 text-sm text-center bg-red-900/20 p-3 rounded-xl border border-red-800/30">{errorMessage}</div>
        )}

        <Button
          onClick={onEnhance}
          disabled={!originalImagePreviewURL || isUpscaling}
          className="w-full bg-white hover:bg-gray-800 text-black h-12 text-base font-semibold rounded-2xl transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {isUpscaling ? (
            <>
              <div className="animate-spin w-4 h-4 border-2 border-black border-t-transparent rounded-full mr-2"></div>
              <span className="text-black">Enhancing...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5 mr-2 text-black" />
              <span className="text-black">Enhance</span>
            </>
          )}
        </Button>

        {hasUpscaledImage && (
          <Button
            onClick={onDownload}
            variant="outline"
            className="w-full border-[#121212] text-gray-300 hover:bg-[#393939] hover:text-white bg-black rounded-xl shadow-lg hover:shadow-xl transition-all duration-200 font-medium"
          >
            <Download className="w-4 h-4 mr-2" />
            Download
          </Button>
        )}
      </div>
    </div>
  );
};
