# backend/app/services/video_upscaling_service.py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, AsyncGenerator
import tempfile
import os
import subprocess
import asyncio
import threading
import queue
import gc
from app.services.super_resolution_service import (
    load_tensorrt_engine_2x, 
    load_tensorrt_engine_4x,
    DynamicTensorRTInfer,
    OPT_INPUT_SHAPE
)
import torchvision.transforms.functional as TF

# Configuration constants matching video_enhancer.py
FRAME_BUFFER_SIZE = 200  # For 2x, use larger buffer
FRAME_BUFFER_SIZE_4X = 100  # For 4x, smaller buffer

# Producer-Consumer pattern for better performance (matching video_enhancer.py)
def producer(video_path: str, frame_queue: queue.Queue):
    """Producer thread that reads frames from video and puts them in queue."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        frame_queue.put(None)
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    cap.release()
    frame_queue.put(None)

def consumer(
    frame_queue: queue.Queue,
    total_frames: int,
    trt_infer: DynamicTensorRTInfer,
    video_properties: dict,
    scale_factor: int,
    progress_callback
):
    """Consumer thread that processes frames and writes output video."""
    temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_video_path.close()
    temp_video_path = temp_video_path.name
    
    final_width = video_properties["width"] * scale_factor
    final_height = video_properties["height"] * scale_factor
    fps = video_properties["fps"]
    
    out_writer = cv2.VideoWriter(temp_video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (final_width, final_height))
    
    batch_size = 2 if scale_factor == 2 else 1
    batch_bgr = []
    frames_processed_count = 0
    
    try:
        while frames_processed_count < total_frames:
            frame_bgr = frame_queue.get()
            if frame_bgr is None:
                break
            batch_bgr.append(frame_bgr)
            
            if len(batch_bgr) == batch_size:
                enhanced_batch = process_batch(trt_infer, batch_bgr, scale_factor)
                for frame in enhanced_batch:
                    out_writer.write(frame)
                frames_processed_count += len(batch_bgr)
                
                # Update progress every 25%
                progress = int((frames_processed_count / total_frames) * 100)
                if progress % 25 == 0 or frames_processed_count == total_frames:
                    progress_callback(progress, f"Processed {frames_processed_count}/{total_frames} frames")
                
                batch_bgr = []
        
        # Process remaining frames with padding (matching video_enhancer.py)
        if batch_bgr:
            original_batch_size = len(batch_bgr)
            missing_frames = batch_size - original_batch_size
            if missing_frames > 0:
                last_frame = batch_bgr[-1]
                for _ in range(missing_frames):
                    batch_bgr.append(last_frame)
            
            enhanced_batch = process_batch(trt_infer, batch_bgr, scale_factor)
            for i in range(original_batch_size):
                out_writer.write(enhanced_batch[i])
            frames_processed_count += original_batch_size
            progress_callback(100, f"Processed {frames_processed_count}/{total_frames} frames")
        
        out_writer.release()
        return temp_video_path
    except Exception as e:
        if out_writer.isOpened():
            out_writer.release()
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        raise e

async def upscale_video_frames(
    video_path: str,
    scale_factor: int,
    mode: str
) -> AsyncGenerator[dict, None]:
    """Upscale video frames using producer-consumer pattern and yield progress updates."""
    
    # Load appropriate TensorRT engine
    trt_infer = None
    if scale_factor == 2 and mode == "Normal":
        trt_infer = load_tensorrt_engine_2x()
    elif scale_factor == 4:
        trt_infer = load_tensorrt_engine_4x()
    else:
        yield {"progress": 0, "error": f"Video upscaling not supported for {scale_factor}x {mode}"}
        return
    
    if trt_infer is None:
        yield {"progress": 0, "error": f"TensorRT engine not available for {scale_factor}x {mode}"}
        return
    
    # Open video to get properties
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        yield {"progress": 0, "error": "Could not open video file"}
        return
    
    video_properties = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    
    total_frames = video_properties["total_frames"]
    frame_buffer_size = FRAME_BUFFER_SIZE if scale_factor == 2 else FRAME_BUFFER_SIZE_4X
    
    yield {"progress": 0, "message": f"Processing {total_frames} frames..."}
    
    # Progress callback function with thread-safe queue
    progress_queue = queue.Queue()
    def progress_callback(progress, message):
        progress_queue.put({"progress": progress, "message": message})
    
    try:
        # Setup producer-consumer pattern
        frame_queue = queue.Queue(maxsize=frame_buffer_size)
        producer_thread = threading.Thread(target=producer, args=(video_path, frame_queue))
        producer_thread.start()
        
        # Run consumer in thread pool to avoid blocking
        import concurrent.futures
        temp_video_path = None
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(
                consumer, frame_queue, total_frames, trt_infer, 
                video_properties, scale_factor, progress_callback
            )
            
            # Yield progress updates as they come
            last_progress = 0
            while not future.done():
                try:
                    # Check for progress updates (non-blocking)
                    while True:
                        try:
                            update = progress_queue.get_nowait()
                            yield update
                            last_progress = update["progress"]
                        except queue.Empty:
                            break
                except:
                    pass
                await asyncio.sleep(0.1)  # Small delay to avoid busy waiting
            
            temp_video_path = future.result()
            
            # Yield any remaining progress updates
            while True:
                try:
                    update = progress_queue.get_nowait()
                    yield update
                except queue.Empty:
                    break
        
        producer_thread.join()
        
        # Extract audio and combine
        yield {"progress": 100, "message": "Combining with audio..."}
        
        # Extract audio
        audio_path = tempfile.NamedTemporaryFile(delete=False, suffix='.aac')
        audio_path_name = audio_path.name
        audio_path.close()
        
        has_audio = False
        try:
            subprocess.run([
                'ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', '-y', audio_path_name
            ], check=True, capture_output=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            has_audio = os.path.exists(audio_path_name) and os.path.getsize(audio_path_name) > 0
        except:
            has_audio = False
        
        # Final output
        final_output = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        final_output_path = final_output.name
        final_output.close()
        
        if has_audio:
            # Use libx264 codec matching video-enhancer-4x.py
            subprocess.run([
                'ffmpeg', '-i', temp_video_path, '-i', audio_path_name,
                '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
                '-c:a', 'aac', '-shortest', '-y', final_output_path
            ], check=True, capture_output=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            if os.path.exists(audio_path_name):
                os.unlink(audio_path_name)
        else:
            # For 2x, use copy codec if no audio (matching video_enhancer.py)
            if scale_factor == 2:
                subprocess.run([
                    'ffmpeg', '-i', temp_video_path, '-c:v', 'copy', '-y', final_output_path
                ], check=True, capture_output=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
            else:
                # For 4x, always re-encode
                subprocess.run([
                    'ffmpeg', '-i', temp_video_path, '-c:v', 'libx264', '-preset', 'medium', 
                    '-crf', '18', '-pix_fmt', 'yuv420p', '-y', final_output_path
                ], check=True, capture_output=True, stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        
        # Cleanup
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        
        yield {"progress": 100, "message": "Video upscaling complete", "output_path": final_output_path}
        
    except Exception as e:
        yield {"progress": 0, "error": str(e)}
        import traceback
        traceback.print_exc()

def process_batch(trt_infer: DynamicTensorRTInfer, frames_bgr: list, scale_factor: int) -> list:
    """Process a batch of frames using TensorRT with optimizations matching video_enhancer.py."""
    if not frames_bgr:
        return []
    
    orig_h, orig_w = frames_bgr[0].shape[:2]
    opt_h, opt_w = OPT_INPUT_SHAPE[2], OPT_INPUT_SHAPE[3]
    
    if scale_factor == 2:
        # 2x processing - matching video_enhancer.py exactly
        # Resize frames BEFORE tensor conversion (optimization)
        preprocessed_frames = [cv2.resize(f, (opt_w, opt_h), interpolation=cv2.INTER_LINEAR) for f in frames_bgr]
        batch_tensor_bgr = torch.from_numpy(np.stack(preprocessed_frames)).permute(0, 3, 1, 2).to('cuda')
        batch_tensor_rgb = batch_tensor_bgr.flip(dims=[1])
        batch_tensor_float = batch_tensor_rgb.float() / 255.0
        batch_tensor_normalized = TF.normalize(batch_tensor_float, [0.5]*3, [0.5]*3)
        
        hr_batch_gpu_large = trt_infer.infer_batch(batch_tensor_normalized)
        
        # Resize to target output size
        target_output_height, target_output_width = orig_h * 2, orig_w * 2
        hr_batch_gpu_resized = F.interpolate(hr_batch_gpu_large, size=(target_output_height, target_output_width), mode='bilinear', align_corners=False)
        
        # Denormalize and convert
        hr_batch_denormalized = torch.clamp((hr_batch_gpu_resized + 1.0) / 2.0, 0, 1) * 255
        hr_batch_uint8 = hr_batch_denormalized.byte()
        hr_batch_cpu = hr_batch_uint8.permute(0, 2, 3, 1).cpu().numpy()
        output_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in hr_batch_cpu]
        
        # Memory cleanup (matching video_enhancer.py)
        del batch_tensor_bgr, batch_tensor_rgb, batch_tensor_float, batch_tensor_normalized
        del hr_batch_gpu_large, hr_batch_gpu_resized, hr_batch_denormalized, hr_batch_uint8, hr_batch_cpu
        gc.collect()
        
    else:  # 4x
        # 4x processing - matching video-enhancer-4x.py exactly
        batch_tensor_bgr = torch.as_tensor(np.stack(frames_bgr), device='cuda')
        batch_tensor_rgb = batch_tensor_bgr.permute(0, 3, 1, 2).flip(dims=[1])
        batch_tensor_float_0_1 = batch_tensor_rgb.float() / 255.0
        
        # Resize on GPU (matching video-enhancer-4x.py)
        batch_tensor_resized = F.interpolate(batch_tensor_float_0_1, size=(opt_h, opt_w), mode='bilinear', align_corners=False)
        batch_tensor_neg1_1 = (batch_tensor_resized * 2.0) - 1.0
        
        hr_batch_gpu_large = trt_infer.infer_batch(batch_tensor_neg1_1)
        
        target_output_height, target_output_width = orig_h * 4, orig_w * 4
        hr_batch_gpu_resized = F.interpolate(hr_batch_gpu_large, size=(target_output_height, target_output_width), mode='bilinear', align_corners=False)
        hr_batch_denormalized = torch.clamp((hr_batch_gpu_resized + 1.0) / 2.0, 0, 1) * 255
        hr_batch_uint8 = hr_batch_denormalized.byte()
        hr_batch_cpu = hr_batch_uint8.permute(0, 2, 3, 1).cpu().numpy()
        output_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in hr_batch_cpu]
        
        # Memory cleanup (matching video-enhancer-4x.py)
        del batch_tensor_bgr, batch_tensor_rgb, batch_tensor_float_0_1, batch_tensor_resized, batch_tensor_neg1_1
        del hr_batch_gpu_large, hr_batch_gpu_resized, hr_batch_denormalized, hr_batch_uint8, hr_batch_cpu
        gc.collect()
    
    return output_frames

