import { useState, useRef, useEffect, useCallback } from 'react';

export interface UseImageZoomPanResult {
    containerRef: React.RefObject<HTMLDivElement>;
    zoom: number;
    panOffset: { x: number; y: number };
    isPanning: boolean;
    handleMouseDown: (event: React.MouseEvent) => void;
}

export function useImageZoomPan(): UseImageZoomPanResult {
    const [zoom, setZoom] = useState(1.0);
    const [isPanning, setIsPanning] = useState(false);
    const [panOffset, setPanOffset] = useState({ x: 0, y: 0 });
    const [lastPanPoint, setLastPanPoint] = useState({ x: 0, y: 0 });
    const containerRef = useRef<HTMLDivElement>(null);

    // Handle mouse wheel for zoom
    useEffect(() => {
        const handleWheel = (event: WheelEvent) => {
            if (containerRef.current && containerRef.current.contains(event.target as Node)) {
                event.preventDefault();
                let newZoom = zoom - event.deltaY * 0.001;
                newZoom = Math.max(0.5, Math.min(5, newZoom));
                setZoom(newZoom);
            }
        };
        window.addEventListener('wheel', handleWheel, { passive: false });
        return () => window.removeEventListener('wheel', handleWheel);
    }, [zoom]);

    const handleMouseDown = (event: React.MouseEvent) => {
        if (zoom > 1.0) {
            setIsPanning(true);
            setLastPanPoint({ x: event.clientX, y: event.clientY });
        }
    };

    const handleMouseMove = useCallback((event: MouseEvent) => {
        if (isPanning && zoom > 1.0) {
            const deltaX = event.clientX - lastPanPoint.x;
            const deltaY = event.clientY - lastPanPoint.y;
            setPanOffset(prev => ({ x: prev.x + deltaX, y: prev.y + deltaY }));
            setLastPanPoint({ x: event.clientX, y: event.clientY });
        }
    }, [isPanning, lastPanPoint, zoom]);

    const handleMouseUp = () => {
        setIsPanning(false);
    };

    useEffect(() => {
        if (isPanning) {
            document.addEventListener('mousemove', handleMouseMove);
            document.addEventListener('mouseup', handleMouseUp);
        }
        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
            document.removeEventListener('mouseup', handleMouseUp);
        };
    }, [isPanning, handleMouseMove]);

    return {
        containerRef,
        zoom,
        panOffset,
        isPanning,
        handleMouseDown,
    };
} 