import { Plus, Trash2, Edit3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';

// Added: Custom CSS for the tooltip slide-in animation.
// This is injected into the document's head for a self-contained component.
const tooltipStyles = `
  @keyframes slideInFromLeft {
    0% {
      opacity: 0;
      transform: translateX(-8px) translateY(-50%);
    }
    100% {
      opacity: 1;
      transform: translateX(0) translateY(-50%);
    }
  }
`;

interface Session {
  id: string;
  name: string;
  imagePreviewURL: string;
  createdAt: string;
}

interface SidebarProps {
  originalImagePreviewURL: string | null;
  onFileUpload: (file: File) => void;
  onDeleteImage: () => void;
  onSessionSelect?: (session: Session) => void;
  activeSessionId?: string | null;
}

export const Sidebar = ({ originalImagePreviewURL, onFileUpload, onDeleteImage, onSessionSelect, activeSessionId }: SidebarProps) => {
  // --- New states for Tooltips ---
  const [showPlusTooltip, setShowPlusTooltip] = useState(false);
  const [showSessionTooltip, setShowSessionTooltip] = useState<string | null>(null);
  const [tooltipPosition, setTooltipPosition] = useState({ x: 0, y: 0 });
  const [sessionTooltipPosition, setSessionTooltipPosition] = useState({ x: 0, y: 0 });

  const [isExpanded, setIsExpanded] = useState(false);
  const [showContextMenu, setShowContextMenu] = useState(false);
  const [contextMenuPosition, setContextMenuPosition] = useState({ x: 0, y: 0 });
  const [isRenaming, setIsRenaming] = useState<string | null>(null);
  const [contextMenuSessionId, setContextMenuSessionId] = useState<string | null>(null);
  const [sessions, setSessions] = useState<Session[]>([]);
  const contextMenuRef = useRef<HTMLDivElement>(null);

  // --- New Refs for Tooltip Positioning ---
  const plusButtonRef = useRef<HTMLButtonElement>(null);
  const sessionThumbnailRefs = useRef<{ [key: string]: HTMLDivElement | null }>({});

  // --- New Ref for Sessions Container ---
  const sessionsContainerRef = useRef<HTMLDivElement>(null);

  // --- New state for active session ---
  const [internalActiveSessionId, setInternalActiveSessionId] = useState<string | null>(null);
  const effectiveActiveSessionId = activeSessionId ?? internalActiveSessionId;

  const generateSessionId = () => {
    return Date.now().toString() + Math.random().toString(36).substr(2, 9);
  };

  // Added: Helper function to format date as seen in the video.
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    const today = new Date();
    const yesterday = new Date(today);
    yesterday.setDate(yesterday.getDate() - 1);

    if (date.toDateString() === today.toDateString()) {
      return 'Today';
    } else if (date.toDateString() === yesterday.toDateString()) {
      return 'Yesterday';
    } else {
      return date.toLocaleDateString();
    }
  };

  const handlePlusClick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'image/*';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
          const imageUrl = event.target?.result as string;
          const newSession: Session = {
            id: generateSessionId(),
            name: `Session ${sessions.length + 1}`,
            imagePreviewURL: imageUrl,
            createdAt: new Date().toISOString()
          };
          setSessions(prev => [newSession, ...prev]);
        };
        reader.readAsDataURL(file);
        onFileUpload(file);
      }
    };
    input.click();
  };

  const handleContextMenu = (e: React.MouseEvent, sessionId: string) => {
    e.preventDefault();
    setContextMenuPosition({ x: e.clientX, y: e.clientY });
    setContextMenuSessionId(sessionId);
    setShowContextMenu(true);
  };

  const handleRename = () => {
    if (contextMenuSessionId) {
      setIsRenaming(contextMenuSessionId);
      setShowContextMenu(false);
    }
  };

  const handleDelete = () => {
    if (contextMenuSessionId) {
      const sessionToDelete = sessions.find(s => s.id === contextMenuSessionId);
      setSessions(prev => prev.filter(session => session.id !== contextMenuSessionId));
      if (sessionToDelete && sessionToDelete.imagePreviewURL === originalImagePreviewURL) {
        onDeleteImage();
      }
      setShowContextMenu(false);
      setContextMenuSessionId(null);
    }
  };

  const handleRenameSubmit = (e: React.FormEvent, sessionId: string, newName: string) => {
    e.preventDefault();
    setSessions(prev =>
      prev.map(session =>
        session.id === sessionId
          ? { ...session, name: newName }
          : session
      )
    );
    setIsRenaming(null);
  };

  const handleRenameCancel = () => {
    setIsRenaming(null);
  };

  const handleClickOutside = (e: MouseEvent) => {
    if (contextMenuRef.current && !contextMenuRef.current.contains(e.target as Node)) {
      setShowContextMenu(false);
    }
  };

  // --- New scroll handling function ---
  const handleScroll = useCallback((e: WheelEvent) => {
    if (sessionsContainerRef.current && isExpanded) {
      e.preventDefault();
      const container = sessionsContainerRef.current;
      const scrollAmount = e.deltaY;

      // Use requestAnimationFrame for smoother performance
      requestAnimationFrame(() => {
        container.scrollTop += scrollAmount;
      });
    }
  }, [isExpanded]);

  // --- New functions for Tooltip logic ---
  const updateTooltipPosition = useCallback((element: HTMLElement | null, setPosition: (pos: { x: number, y: number }) => void) => {
    if (element) {
      const rect = element.getBoundingClientRect();
      // Position tooltip to the right of the element with a small gap.
      setPosition({
        x: rect.right + 12,
        y: rect.top + rect.height / 2
      });
    }
  }, []);

  const handlePlusMouseEnter = useCallback(() => {
    setShowPlusTooltip(true);
    updateTooltipPosition(plusButtonRef.current, setTooltipPosition);
  }, [updateTooltipPosition]);

  const handleSessionMouseEnter = useCallback((sessionId: string) => {
    setShowSessionTooltip(sessionId);
    updateTooltipPosition(sessionThumbnailRefs.current[sessionId], setSessionTooltipPosition);
  }, [updateTooltipPosition]);

  // --- Session click handler ---
  const handleSessionClick = (session: Session) => {
    setInternalActiveSessionId(session.id);
    if (onSessionSelect) onSessionSelect(session);
  };

  // --- useEffect to inject tooltip animation styles and add scroll listener ---
  useEffect(() => {
    document.addEventListener('click', handleClickOutside);

    // Inject custom styles for tooltip animation
    const style = document.createElement('style');
    style.textContent = tooltipStyles;
    document.head.appendChild(style);

    // Add scroll listener to sessions container
    const container = sessionsContainerRef.current;
    if (container) {
      container.addEventListener('wheel', handleScroll, { passive: false });
    }

    return () => {
      document.removeEventListener('click', handleClickOutside);
      if (document.head.contains(style)) {
        document.head.removeChild(style);
      }
      if (container) {
        container.removeEventListener('wheel', handleScroll);
      }
    };
  }, [handleScroll]);

  useEffect(() => {
    if (originalImagePreviewURL && sessions.length === 0) {
      const initialSession: Session = {
        id: generateSessionId(),
        name: 'Initial Session',
        imagePreviewURL: originalImagePreviewURL,
        createdAt: new Date().toISOString()
      };
      setSessions([initialSession]);
    }
  }, [originalImagePreviewURL, sessions.length]);

  return (
    <div className="flex items-center justify-center min-h-screen">
      <div
        className={`group bg-[#121212] bg-gradient-to-b from-gray-950/98 to-black/95 border border-[#121212] border-l-0 flex flex-col backdrop-blur-2xl h-[400px] relative left-[-7px] align-center rounded-xl transition-all duration-200 ease-in-out ${isExpanded ? 'w-20' : 'w-6'}`}
        onMouseEnter={() => setIsExpanded(true)}
        onMouseLeave={() => setIsExpanded(false)}
        style={{ minWidth: isExpanded ? '5rem' : '1.5rem', maxWidth: isExpanded ? '5rem' : '1.5rem' }}
      >
        <div className="absolute inset-0 bg-gradient-to-r from-transparent via-gray-700/5 to-transparent pointer-events-none" />

        <div className={`p-4 relative flex flex-col items-center transition-all duration-300 ease-in-out ${isExpanded ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-6 pointer-events-none'}`}>
          <div className="relative">
            <Button
              ref={plusButtonRef} // Attach ref
              onClick={handlePlusClick}
              variant="outline"
              className="w-12 h-12 border-0 bg-white hover:bg-white text-gray-400 hover:text-white transition-all duration-300 shadow-inner hover:shadow backdrop-blur-sm hover:scale-110 relative overflow-hidden group"
              // Add tooltip event handlers
              onMouseEnter={handlePlusMouseEnter}
              onMouseLeave={() => setShowPlusTooltip(false)}
            >
              <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
              <Plus className="w-5 h-5 relative z-10 text-black" />
            </Button>
          </div>
        </div>

        <div className={`w-10 mx-auto h-px bg-gradient-to-r from-transparent via-gray-600 to-transparent my-2 transition-opacity duration-300 ${isExpanded ? 'opacity-100' : 'opacity-0'}`} />

        <div
          ref={sessionsContainerRef}
          className={`flex-1 p-4 overflow-y-scroll flex flex-col items-center transition-all duration-300 ease-in-out ${isExpanded ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-6 pointer-events-none'}`}
          style={{
            scrollbarWidth: 'none',
            msOverflowStyle: 'none',
            scrollBehavior: 'auto'
          }}
        >
          <style>{`div::-webkit-scrollbar { display: none; }`}</style>
          <div className="space-y-4">
            {sessions.map((session) => (
              <div
                key={session.id}
                // Attach ref for each session
                ref={(el) => { sessionThumbnailRefs.current[session.id] = el; }}
                className={`relative group w-12 h-12 rounded border border-gray-700/30 hover:border-gray-600/50 overflow-hidden shadow-lg transition-all duration-300 hover:scale-110 bg-gray-800/20 backdrop-blur-sm cursor-pointer flex-shrink-0 ${effectiveActiveSessionId === session.id ? 'ring-2 ring-white' : ''}`}
                onClick={() => handleSessionClick(session)}
                // Add tooltip and context menu event handlers
                onMouseEnter={() => handleSessionMouseEnter(session.id)}
                onMouseLeave={() => setShowSessionTooltip(null)}
                onContextMenu={(e) => handleContextMenu(e, session.id)}
              >
                <div className="absolute inset-0 bg-gradient-to-br from-white/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
                <img src={session.imagePreviewURL} alt="Session thumbnail" className="w-full h-full object-cover relative z-10" />
              </div>
            ))}
          </div>
        </div>

        <div className={`p-4 transition-opacity duration-300 ${isExpanded ? 'opacity-100' : 'opacity-0'}`} />
      </div>

      {/* --- Tooltip Rendering using React Portals --- */}

      {/* Plus Button Tooltip */}
      {showPlusTooltip && isExpanded && createPortal(
        <div
          className="fixed z-50 bg-[#383838] text-white px-3 py-2 rounded-lg shadow-2xl font-sans text-sm font-medium flex items-center min-w-[120px] border border-gray-700/40 backdrop-blur-sm"
          style={{
            left: tooltipPosition.x,
            top: tooltipPosition.y,
            transform: 'translateY(-50%)', // Vertically center on cursor
            animation: 'slideInFromLeft 0.2s ease-out forwards'
          }}
        >
          <Plus className="w-4 h-4 mr-2" /> New Session
        </div>,
        document.body
      )}

      {/* Session Thumbnails Tooltip */}
      {showSessionTooltip && isExpanded && sessions.find(s => s.id === showSessionTooltip) && createPortal(
        <div
          className="fixed z-50 bg-[#383838] text-white px-3 py-2 rounded-lg shadow-2xl font-sans text-sm min-w-[120px] border border-gray-700/40 backdrop-blur-sm"
          style={{
            left: sessionTooltipPosition.x,
            top: sessionTooltipPosition.y,
            transform: 'translateY(-50%)', // Vertically center
            animation: 'slideInFromLeft 0.2s ease-out forwards'
          }}
        >
          <div className="font-medium text-white">{sessions.find(s => s.id === showSessionTooltip)?.name}</div>
          <div className="text-xs text-gray-400 mt-1">{formatDate(sessions.find(s => s.id === showSessionTooltip)!.createdAt)}</div>
        </div>,
        document.body
      )}

      {/* Context Menu */}
      {showContextMenu && (
        <div ref={contextMenuRef} className="fixed z-50 bg-[#383838] border border-gray-700/40 rounded-lg shadow-2xl backdrop-blur-sm min-w-[140px]" style={{ left: contextMenuPosition.x, top: contextMenuPosition.y }}>
          <button onClick={handleRename} className="w-full px-4 py-2 text-left text-white hover:bg-gray-800/50 flex items-center gap-2 transition-colors duration-200 first:rounded-t-lg last:rounded-b-lg">
            <Edit3 className="w-4 h-4" /> Rename
          </button>
          <button onClick={handleDelete} className="w-full px-4 py-2 text-left text-red-400 hover:bg-red-500/20 flex items-center gap-2 transition-colors duration-200 first:rounded-t-lg last:rounded-b-lg">
            <Trash2 className="w-4 h-4" /> Delete
          </button>
        </div>
      )}

      {/* Rename Modal */}
      {isRenaming && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-gray-900/95 border border-gray-700/40 rounded-lg p-6 backdrop-blur-sm min-w-[300px]">
            <h3 className="text-white font-medium mb-4">Rename Session</h3>
            <form onSubmit={(e) => {
              const newName = (e.currentTarget.elements.namedItem('sessionName') as HTMLInputElement).value;
              handleRenameSubmit(e, isRenaming, newName);
            }}>
              <input type="text" name="sessionName" defaultValue={sessions.find(s => s.id === isRenaming)?.name || ''} className="w-full px-3 py-2 bg-gray-800/50 border border-gray-600/50 rounded-md text-white placeholder-gray-400 focus:outline-none focus:border-gray-500 focus:ring-1 focus:ring-gray-500" placeholder="Enter session name" autoFocus />
              <div className="flex gap-2 mt-4">
                <button type="submit" className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-md transition-colors duration-200">Save</button>
                <button type="button" onClick={handleRenameCancel} className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-md transition-colors duration-200">Cancel</button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};