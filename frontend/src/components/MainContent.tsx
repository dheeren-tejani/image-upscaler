import { Upload } from 'lucide-react';
import { Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ImageComparison } from './ImageComparison';
import { useImageZoomPan } from './useImageZoomPan';
import { useRef, useEffect } from 'react';

interface MainContentProps {
  originalImagePreviewURL: string | null;
  upscaledImageBase64: string | null;
  onFileUpload: (file: File) => void;
  isUpscaling: boolean;
  isVideoMode?: boolean;
  videoFile?: File | null;
  videoPreviewURL?: string | null;
  upscaledVideoURL?: string | null;
  videoProgress?: number;
  videoProgressMessage?: string;
  onVideoUpload?: (file: File) => void;
  onVideoUpscale?: () => void;
  onDownloadVideo?: () => void;
}

export const MainContent = ({
  originalImagePreviewURL,
  upscaledImageBase64,
  onFileUpload,
  isUpscaling,
  isVideoMode = false,
  videoFile,
  videoPreviewURL,
  upscaledVideoURL,
  videoProgress = 0,
  videoProgressMessage = '',
  onVideoUpload,
  onVideoUpscale,
  onDownloadVideo
}: MainContentProps) => {
  const handleFileInputChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (isVideoMode && onVideoUpload) {
        onVideoUpload(file);
      } else {
        onFileUpload(file);
      }
    }
  };

  const handleUploadClick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = isVideoMode ? 'video/*' : 'image/*';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        if (isVideoMode && onVideoUpload) {
          onVideoUpload(file);
        } else {
          onFileUpload(file);
        }
      }
    };
    input.click();
  };

  // Move hooks to top level - always call them
  const {
    containerRef,
    zoom,
    panOffset,
    isPanning,
    handleMouseDown,
  } = useImageZoomPan();

  // Animation ref for original image
  const origImgRef = useRef<HTMLImageElement>(null);

  // Animation ref for enhanced image
  const enhancedImgRef = useRef<HTMLImageElement>(null);

  useEffect(() => {
    if (origImgRef.current) {
      origImgRef.current.classList.remove('fadein-appear');
      void origImgRef.current.offsetWidth;
      origImgRef.current.classList.add('fadein-appear');
    }
  }, [originalImagePreviewURL]);

  useEffect(() => {
    if (enhancedImgRef.current) {
      enhancedImgRef.current.classList.remove('fadein-appear');
      // Force reflow to restart animation
      void enhancedImgRef.current.offsetWidth;
      enhancedImgRef.current.classList.add('fadein-appear');
    }
  }, [upscaledImageBase64]);

  const simpleSparkleAnimation = `
  @keyframes sparkle-rotate {
    0% { 
      opacity: 0.5;
      transform: scale(0.9) rotate(0deg);
    }
    50% { 
      opacity: 1;
      transform: scale(1.1) rotate(180deg);
    }
    100% { 
      opacity: 0.5;
      transform: scale(0.9) rotate(360deg);
    }
  }
`;

  const SimpleSparklesLoading = () => {
    return (
      <>
        <style>{simpleSparkleAnimation}</style>
        <div className="relative flex items-center justify-center mb-8">
          <span className="absolute inline-flex h-full w-full rounded-full bg-gradient-to-br from-white/30 via-white/20 to-white/10 animate-pulse"></span>

          <Sparkles
            className="h-16 w-16 text-white"
            style={{
              animation: 'sparkle-rotate 2s ease-in-out infinite'
            }}
          />
        </div>
      </>
    );
  };

  // Complex multi-sparkle animation
  const complexSparkleAnimation = `
  @keyframes sparkle {
    0%, 100% { 
      opacity: 0.3;
      transform: scale(0.8) rotate(0deg);
    }
    50% { 
      opacity: 1;
      transform: scale(1.2) rotate(180deg);
    }
  }
  
  @keyframes sparkle-delayed {
    0%, 100% { 
      opacity: 0.5;
      transform: scale(0.9) rotate(0deg);
    }
    50% { 
      opacity: 1;
      transform: scale(1.1) rotate(-180deg);
    }
  }
  
  @keyframes sparkle-pulse {
    0%, 100% { 
      opacity: 0.4;
      transform: scale(1);
    }
    50% { 
      opacity: 1;
      transform: scale(1.3);
    }
  }
`;

  const ComplexSparklesLoading = () => {
    return (
      <>
        <style>{complexSparkleAnimation}</style>
        <div className="relative flex items-center justify-center mb-8">
          <span className="absolute inline-flex h-full w-full rounded-full bg-gradient-to-br from-white/30 via-white/20 to-white/10 animate-pulse"></span>

          {/* Multiple sparkles with different animations */}
          <div className="relative flex items-center justify-center">
            <Sparkles
              className="h-16 w-16 text-white absolute"
              style={{
                animation: 'sparkle 2s ease-in-out infinite'
              }}
            />
            <Sparkles
              className="h-12 w-12 text-white/70 absolute"
              style={{
                animation: 'sparkle-delayed 2.5s ease-in-out infinite 0.5s'
              }}
            />
            <Sparkles
              className="h-8 w-8 text-white/50 absolute"
              style={{
                animation: 'sparkle-pulse 1.8s ease-in-out infinite 1s'
              }}
            />
          </div>
        </div>
      </>
    );
  };

  // Circular progress component for video
  const CircularProgress = ({ progress }: { progress: number }) => {
    const radius = 60;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference - (progress / 100) * circumference;

    return (
      <div className="relative w-32 h-32">
        <svg className="transform -rotate-90 w-32 h-32">
          <circle
            cx="64"
            cy="64"
            r={radius}
            stroke="rgba(255, 255, 255, 0.1)"
            strokeWidth="8"
            fill="none"
          />
          <circle
            cx="64"
            cy="64"
            r={radius}
            stroke="white"
            strokeWidth="8"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className="transition-all duration-300"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-2xl font-bold text-white">{progress}%</span>
        </div>
      </div>
    );
  };

  // Video mode rendering
  if (isVideoMode) {
    if (upscaledVideoURL && videoPreviewURL) {
      return (
        <div className="flex-1 bg-black flex items-center justify-center h-screen overflow-hidden">
          <div className="w-full h-full flex flex-col items-center justify-center p-8 space-y-6">
            <div className="grid grid-cols-2 gap-8 w-full max-w-6xl">
              <div className="space-y-4">
                <h3 className="text-white text-xl font-semibold">Original</h3>
                <video
                  src={videoPreviewURL}
                  controls
                  className="w-full rounded-lg"
                />
              </div>
              <div className="space-y-4">
                <h3 className="text-white text-xl font-semibold">Upscaled</h3>
                <video
                  src={upscaledVideoURL}
                  controls
                  className="w-full rounded-lg"
                />
              </div>
            </div>
            {onDownloadVideo && (
              <button
                onClick={onDownloadVideo}
                className="bg-white hover:bg-gray-100 text-black px-6 py-3 rounded-lg font-semibold transition-colors"
              >
                Download Upscaled Video
              </button>
            )}
          </div>
        </div>
      );
    }

    if (isUpscaling && videoFile) {
      return (
        <div className="flex-1 bg-black flex items-center justify-center h-screen overflow-hidden">
          <div className="flex flex-col items-center justify-center space-y-6">
            <CircularProgress progress={videoProgress} />
            <h2 className="text-2xl font-bold text-white mb-2 font-sans">Upscaling video...</h2>
            <p className="text-gray-400 font-sans">{videoProgressMessage || `Progress: ${videoProgress}%`}</p>
          </div>
        </div>
      );
    }

    if (videoPreviewURL) {
      return (
        <div className="flex-1 bg-black flex items-center justify-center h-screen overflow-hidden">
          <div className="w-full h-full flex flex-col items-center justify-center p-8 space-y-6">
            <video
              src={videoPreviewURL}
              controls
              className="max-w-4xl max-h-[80vh] rounded-lg"
            />
            {onVideoUpscale && (
              <button
                onClick={onVideoUpscale}
                className="bg-white hover:bg-gray-100 text-black px-8 py-4 text-lg font-semibold rounded-2xl shadow-lg transition-all"
              >
                Upscale Video
              </button>
            )}
          </div>
        </div>
      );
    }
  }

  if (upscaledImageBase64 && originalImagePreviewURL) {
    return (
      <div className="flex-1 bg-black flex items-center justify-center h-screen overflow-hidden">
        <style>{`
          .fadein-appear {
            animation: fadeinScale 0.7s cubic-bezier(0.4,0,0.2,1);
          }
          @keyframes fadeinScale {
            0% { opacity: 0; transform: scale(0.97); }
            100% { opacity: 1; transform: scale(1); }
          }
        `}</style>
        <div className="w-full h-full flex items-center justify-center max-h-[90vh] max-w-[90vw] mx-auto">
          <ImageComparison
            originalImage={originalImagePreviewURL}
            enhancedImage={`data:image/png;base64,${upscaledImageBase64}`}
            enhancedImgRef={enhancedImgRef}
          />
        </div>
      </div>
    );
  }

  if (originalImagePreviewURL) {
    if (isUpscaling) {
      return (
        <div className="flex-1 bg-black flex items-center justify-center h-screen overflow-hidden">
          <div className="flex flex-col items-center justify-center w-full h-full">
            <ComplexSparklesLoading />
            <h2 className="text-2xl font-bold text-white mb-2 font-sans">Enhancing your image...</h2>
            <p className="text-gray-400 font-sans">This may take a few moments</p>
          </div>
        </div>
      );
    }
    return (
      <div className="flex-1 bg-black flex items-center justify-center h-screen overflow-hidden">
        <style>{`
          .fadein-appear {
            animation: fadeinScale 0.7s cubic-bezier(0.4,0,0.2,1);
          }
          @keyframes fadeinScale {
            0% { opacity: 0; transform: scale(0.97); }
            100% { opacity: 1; transform: scale(1); }
          }
        `}</style>
        <div className="relative flex items-center justify-center w-full h-full max-h-[90vh] max-w-[90vw] mx-auto">
          <div
            ref={containerRef}
            className="flex items-center justify-center max-w-[80vw] max-h-[80vh] rounded-3xl overflow-hidden shadow-2xl select-none bg-black backdrop-blur-sm border border-gray-700/20"
            style={{
              transform: `scale(${zoom}) translate(${panOffset.x / zoom}px, ${panOffset.y / zoom}px)`,
              transition: isPanning ? 'none' : 'transform 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
              cursor: zoom > 1.0 ? (isPanning ? 'grabbing' : 'grab') : 'default',
              boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.8), 0 0 0 1px rgba(255, 255, 255, 0.05)'
            }}
            onMouseDown={handleMouseDown}
          >
            <img
              ref={origImgRef}
              src={originalImagePreviewURL}
              alt="Uploaded image"
              className="block object-contain max-w-[80vw] max-h-[80vh] rounded-2xl fadein-appear"
              draggable={false}
              style={{ margin: '0 auto' }}
            />
          </div>
          {/* Zoom percentage overlay */}
          <div className="absolute -bottom-6 right-4 z-20">
            <div className="bg-[#383838] rounded px-3 py-1 shadow flex items-center justify-center">
              <span className="text-sm font-semibold text-white font-sans">{Math.round(zoom * 100)}%</span>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 bg-black flex items-center justify-center h-screen overflow-hidden">
      <div className="text-center max-w-2xl w-full flex flex-col items-center justify-center">
        {/* Abstract illustration placeholder */}
        <div className="mb-8 flex justify-center space-x-4">
          <div className="w-16 h-20 bg-gradient-to-br from-gray-800/60 to-gray-700/60 rounded-2xl border border-gray-700/50 shadow-lg backdrop-blur-sm"></div>
          <div className="w-16 h-20 bg-gradient-to-br from-white/40 to-white/40 rounded-2xl border border-white/30 shadow-lg backdrop-blur-sm"></div>
          <div className="w-16 h-20 bg-gradient-to-br from-gray-800/60 to-gray-700/60 rounded-2xl border border-gray-700/50 shadow-lg backdrop-blur-sm"></div>
        </div>

        <h1 className="text-6xl font-bold text-white mb-4 bg-gradient-to-r from-white to-gray-300 bg-clip-text text-transparent font-sans">
          Enhancer
        </h1>

        <p className="text-xl text-gray-400 mb-8 leading-relaxed font-sans">
          Upscale and generate new details for {isVideoMode ? 'videos' : 'images'} up to{' '}
          <span className="text-white font-semibold">4K</span>{' '}
        </p>

        <div className="space-y-4 w-full flex flex-col items-center justify-center">
          <Button
            onClick={handleUploadClick}
            className="bg-white hover:bg-white text-black px-8 py-4 text-lg font-semibold rounded-2xl shadow-lg shadow-white/25 hover:shadow-xl hover:shadow-white/30 transition-all duration-200 font-sans"
          >
            <Upload className="w-6 h-6 mr-3" />
            Upload
          </Button>

          <Button
            variant="ghost"
            className="block mx-auto text-gray-400 hover:text-white transition-colors duration-200 font-medium font-sans"
            disabled
          >
            Select asset
          </Button>
        </div>

        <p className="text-sm text-gray-500 mt-6 font-medium font-sans">
          Max 2.5MB / 10 - 15 seconds processing time
        </p>
      </div>
    </div>
  );
};