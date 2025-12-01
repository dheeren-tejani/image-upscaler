import { useState } from 'react';
import { Sidebar } from '@/components/Sidebar';
import { MainContent } from '@/components/MainContent';
import { ControlPanel } from '@/components/ControlPanel';

const Index = () => {
  const [originalImageFile, setOriginalImageFile] = useState<File | null>(null);
  const [originalImagePreviewURL, setOriginalImagePreviewURL] = useState<string | null>(null);
  const [upscaledImageBase64, setUpscaledImageBase64] = useState<string | null>(null);
  const [isUpscaling, setIsUpscaling] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [selectedUpscaleFactor, setSelectedUpscaleFactor] = useState(2);
  const [selectedMode, setSelectedMode] = useState<'Normal' | 'Crazy'>('Normal');
  const [inputImageInfo, setInputImageInfo] = useState<{ width: number; height: number; size: number } | null>(null);
  const [enhancedImageInfo, setEnhancedImageInfo] = useState<{ width: number; height: number; size: number } | null>(null);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  // Video upscaling state
  const [isVideoMode, setIsVideoMode] = useState(false);
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [videoPreviewURL, setVideoPreviewURL] = useState<string | null>(null);
  const [upscaledVideoURL, setUpscaledVideoURL] = useState<string | null>(null);
  const [videoProgress, setVideoProgress] = useState(0);
  const [videoProgressMessage, setVideoProgressMessage] = useState('');

  const handleFileUpload = (file: File) => {
    setOriginalImageFile(file);
    setOriginalImagePreviewURL(URL.createObjectURL(file));
    setUpscaledImageBase64(null);
    setErrorMessage(null);
    // Get image dimensions
    const img = new window.Image();
    img.onload = () => {
      setInputImageInfo({ width: img.width, height: img.height, size: file.size });
    };
    img.src = URL.createObjectURL(file);
  };

  const handleDeleteImage = () => {
    setOriginalImageFile(null);
    setOriginalImagePreviewURL(null);
    setUpscaledImageBase64(null);
    setErrorMessage(null);
    setInputImageInfo(null);
    setEnhancedImageInfo(null);
  };

  const handleEnhance = async () => {
    if (!originalImageFile) return;

    setIsUpscaling(true);
    setErrorMessage(null);

    try {
      const formData = new FormData();
      formData.append('file', originalImageFile);
      formData.append('scale_factor', selectedUpscaleFactor.toString());

      // Add mode parameter for 2x upscaling
      if (selectedUpscaleFactor === 2) {
        formData.append('mode', selectedMode);
      }

      const response = await fetch('http://127.0.0.1:8000/api/v1/sr/upscale-image', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      setUpscaledImageBase64(result.upscaled_image_base64);
      // Get enhanced image info
      const img = new window.Image();
      img.onload = () => {
        // Calculate base64 size in bytes
        const base64Length = result.upscaled_image_base64.length;
        const size = Math.floor((base64Length * 3) / 4 - (result.upscaled_image_base64.endsWith('==') ? 2 : result.upscaled_image_base64.endsWith('=') ? 1 : 0));
        setEnhancedImageInfo({ width: img.width, height: img.height, size });
      };
      img.src = `data:image/png;base64,${result.upscaled_image_base64}`;
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'An error occurred');
    } finally {
      setIsUpscaling(false);
    }
  };

  const handleDownload = () => {
    if (!upscaledImageBase64) return;

    const link = document.createElement('a');
    link.href = `data:image/png;base64,${upscaledImageBase64}`;
    link.download = `enhanced-image-${selectedUpscaleFactor}x.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleVideoUpload = (file: File) => {
    setVideoFile(file);
    setVideoPreviewURL(URL.createObjectURL(file));
    setUpscaledVideoURL(null);
    setVideoProgress(0);
    setVideoProgressMessage('');
    setErrorMessage(null);
  };

  const handleVideoUpscale = async () => {
    if (!videoFile) return;

    setIsUpscaling(true);
    setVideoProgress(0);
    setVideoProgressMessage('Starting video upscaling...');
    setErrorMessage(null);

    try {
      const formData = new FormData();
      formData.append('file', videoFile);
      formData.append('scale_factor', selectedUpscaleFactor.toString());
      if (selectedUpscaleFactor === 2) {
        formData.append('mode', selectedMode);
      }

      const response = await fetch('http://127.0.0.1:8000/api/v1/sr/upscale-video', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // Handle Server-Sent Events
      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        throw new Error('Response body is not readable');
      }

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const data = JSON.parse(line.slice(6));

            if (data.error) {
              throw new Error(data.error);
            }

            if (data.progress !== undefined) {
              setVideoProgress(data.progress);
            }

            if (data.message) {
              setVideoProgressMessage(data.message);
            }

            if (data.video_base64) {
              // Create blob URL from base64
              const videoBlob = base64ToBlob(data.video_base64, 'video/mp4');
              const videoURL = URL.createObjectURL(videoBlob);
              setUpscaledVideoURL(videoURL);
              setVideoProgress(100);
              setVideoProgressMessage('Video upscaling complete!');
            }
          }
        }
      }
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : 'An error occurred');
    } finally {
      setIsUpscaling(false);
    }
  };

  const base64ToBlob = (base64: string, mimeType: string): Blob => {
    const byteCharacters = atob(base64);
    const byteNumbers = new Array(byteCharacters.length);
    for (let i = 0; i < byteCharacters.length; i++) {
      byteNumbers[i] = byteCharacters.charCodeAt(i);
    }
    const byteArray = new Uint8Array(byteNumbers);
    return new Blob([byteArray], { type: mimeType });
  };

  const handleDownloadVideo = () => {
    if (!upscaledVideoURL) return;
    const link = document.createElement('a');
    link.href = upscaledVideoURL;
    link.download = `enhanced-video-${selectedUpscaleFactor}x.mp4`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // Handle session selection from Sidebar
  const handleSessionSelect = (session: { id: string; imagePreviewURL: string; }) => {
    setActiveSessionId(session.id);
    setOriginalImagePreviewURL(session.imagePreviewURL);
    setUpscaledImageBase64(null);
    setErrorMessage(null);
    setEnhancedImageInfo(null);
    setOriginalImageFile(null); // Optionally clear file

    // Dynamically calculate and update inputImageInfo for the new image
    const img = new window.Image();
    img.onload = () => {
      // Try to estimate file size from base64 data URL if possible
      let size = 0;
      if (session.imagePreviewURL.startsWith('data:image/')) {
        // Remove the data URL prefix
        const base64 = session.imagePreviewURL.split(',')[1] || '';
        size = Math.floor((base64.length * 3) / 4 - (base64.endsWith('==') ? 2 : base64.endsWith('=') ? 1 : 0));
      }
      setInputImageInfo({ width: img.width, height: img.height, size });
    };
    img.src = session.imagePreviewURL;
  };

  return (
    <div className="h-screen w-full bg-black text-white flex overflow-hidden relative">
      <div className="absolute left-0 top-0 h-full z-30">
        <Sidebar
          originalImagePreviewURL={originalImagePreviewURL}
          onFileUpload={handleFileUpload}
          onDeleteImage={handleDeleteImage}
          onSessionSelect={handleSessionSelect}
          activeSessionId={activeSessionId}
        />
      </div>

      <div className="flex-1 h-screen overflow-y-auto ml-20">
        <MainContent
          originalImagePreviewURL={originalImagePreviewURL}
          upscaledImageBase64={upscaledImageBase64}
          onFileUpload={handleFileUpload}
          isUpscaling={isUpscaling}
          isVideoMode={isVideoMode}
          videoFile={videoFile}
          videoPreviewURL={videoPreviewURL}
          upscaledVideoURL={upscaledVideoURL}
          videoProgress={videoProgress}
          videoProgressMessage={videoProgressMessage}
          onVideoUpload={handleVideoUpload}
          onVideoUpscale={handleVideoUpscale}
          onDownloadVideo={handleDownloadVideo}
        />
      </div>

      <ControlPanel
        originalImagePreviewURL={originalImagePreviewURL}
        selectedUpscaleFactor={selectedUpscaleFactor}
        onUpscaleFactorChange={setSelectedUpscaleFactor}
        selectedMode={selectedMode}
        onModeChange={setSelectedMode}
        onEnhance={isVideoMode ? handleVideoUpscale : handleEnhance}
        onDownload={isVideoMode ? handleDownloadVideo : handleDownload}
        isUpscaling={isUpscaling}
        hasUpscaledImage={isVideoMode ? !!upscaledVideoURL : !!upscaledImageBase64}
        errorMessage={errorMessage}
        inputImageInfo={inputImageInfo}
        enhancedImageInfo={enhancedImageInfo}
        isVideoMode={isVideoMode}
        onVideoModeChange={setIsVideoMode}
      />
    </div>
  );
};

export default Index;
