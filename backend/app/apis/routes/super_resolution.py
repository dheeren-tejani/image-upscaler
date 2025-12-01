# backend/app/apis/routes/super_resolution.py
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse
from app.schemas.super_resolution import SRResponse
from app.services import super_resolution_service # Import the service module
from app.services import video_upscaling_service
import os
import json
import tempfile
import asyncio

router = APIRouter()

# Security constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB limit
MAX_VIDEO_SIZE = 500 * 1024 * 1024  # 500MB limit for videos
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp'}
ALLOWED_MIME_TYPES = {'image/jpeg', 'image/png', 'image/webp'}
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
ALLOWED_VIDEO_MIME_TYPES = {'video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska', 'video/webm'}

def validate_file(file: UploadFile) -> None:
    """Validate uploaded file for security"""
    # Check file size
    if file.size and file.size > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB")
    
    # Check file extension
    if file.filename:
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Invalid file extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")
    
    # Check MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image (PNG, JPG, WEBP)")

@router.post("/upscale-image", response_model=SRResponse)
async def upscale_image_endpoint(
    file: UploadFile = File(...),
    scale_factor: int = Form(default=2, description="Upscale factor (2, 4, or 8)"),
    mode: str = Form(default="Normal", description="Mode for 2x upscaling (Normal or Crazy)")
):
    # Security validation
    validate_file(file)
    
    # Validate scale factor
    if scale_factor not in [2, 4, 8]:
        raise HTTPException(status_code=400, detail="Scale factor must be either 2, 4, or 8")
    
    # Validate mode for 2x upscaling
    if scale_factor == 2 and mode not in ["Normal", "Crazy"]:
        raise HTTPException(status_code=400, detail="Mode must be either 'Normal' or 'Crazy' for 2x upscaling")
    
    # Get the model instance via the service getter. This also attempts to load if None.
    current_sr_model = super_resolution_service.get_sr_model_instance(scale_factor, mode) 
    
    if current_sr_model is None:
        print(f"SR_ENDPOINT: {scale_factor}x {mode} SR Model could not be loaded/retrieved by service. Check server logs for load errors.")
        raise HTTPException(status_code=503, detail=f"{scale_factor}x {mode} Super-Resolution model is not available or failed to load. Please try again later or check server logs.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    
    # Additional file content validation
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB")
        
    try:
        print(f"SR_ENDPOINT: Received image '{file.filename}' for {scale_factor}x {mode} upscaling, size: {len(image_bytes)} bytes.")
        # upscale_image_bytes will now use the model instance obtained via get_sr_model_instance()
        base64_upscaled_image = await super_resolution_service.upscale_image_bytes(image_bytes, scale_factor, mode)
        
        if base64_upscaled_image is None:
            raise HTTPException(status_code=500, detail="Failed to upscale image due to an internal processing error (service returned None).")
        if base64_upscaled_image == "MODEL_NOT_LOADED_ERROR":
             raise HTTPException(status_code=503, detail=f"{scale_factor}x {mode} Super-Resolution model is not loaded or failed to initialize.")
        if base64_upscaled_image == "CUDA_OOM_ERROR":
            raise HTTPException(status_code=507, detail="CUDA out of memory during upscaling. Image may be too large.")

        return SRResponse(upscaled_image_base64=base64_upscaled_image, message=f"Image upscaled successfully by {scale_factor}x ({mode})")
    
    except HTTPException as http_exc:
        raise http_exc
    except RuntimeError as e: 
        print(f"SR_ENDPOINT: Unhandled runtime error during upscale: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Runtime error during image upscaling: {str(e)}")
    except Exception as e:
        print(f"SR_ENDPOINT: Unexpected error during upscale: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during upscaling: {str(e)}")

@router.post("/upscale-video")
async def upscale_video_endpoint(
    file: UploadFile = File(...),
    scale_factor: int = Form(default=2, description="Upscale factor (2 or 4)"),
    mode: str = Form(default="Normal", description="Mode for 2x upscaling (Normal or Crazy)")
):
    """Upscale video with progress tracking via Server-Sent Events."""
    # Validate scale factor
    if scale_factor not in [2, 4]:
        raise HTTPException(status_code=400, detail="Scale factor must be either 2 or 4 for video upscaling")
    
    # Validate mode for 2x upscaling
    if scale_factor == 2 and mode not in ["Normal", "Crazy"]:
        raise HTTPException(status_code=400, detail="Mode must be either 'Normal' or 'Crazy' for 2x upscaling")
    
    # Validate file
    if file.filename:
        file_ext = os.path.splitext(file.filename.lower())[1]
        if file_ext not in ALLOWED_VIDEO_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"Invalid video file extension. Allowed: {', '.join(ALLOWED_VIDEO_EXTENSIONS)}")
    
    if file.content_type not in ALLOWED_VIDEO_MIME_TYPES:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a video file")
    
    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")
    
    if len(video_bytes) > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=413, detail=f"Video too large. Maximum size is {MAX_VIDEO_SIZE // (1024*1024)}MB")
    
    # Save uploaded video to temp file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1] if file.filename else '.mp4')
    temp_video.write(video_bytes)
    temp_video.close()
    
    async def generate_progress():
        """Generate Server-Sent Events for progress updates."""
        try:
            final_output_path = None
            async for update in video_upscaling_service.upscale_video_frames(
                temp_video.name, scale_factor, mode
            ):
                if "error" in update:
                    yield f"data: {json.dumps(update)}\n\n"
                    break
                elif "output_path" in update:
                    final_output_path = update["output_path"]
                    # Read the output video and send as base64
                    with open(final_output_path, 'rb') as f:
                        import base64
                        video_base64 = base64.b64encode(f.read()).decode('utf-8')
                    update["video_base64"] = video_base64
                    yield f"data: {json.dumps(update)}\n\n"
                else:
                    yield f"data: {json.dumps(update)}\n\n"
            
            # Cleanup
            if os.path.exists(temp_video.name):
                os.unlink(temp_video.name)
            if final_output_path and os.path.exists(final_output_path):
                os.unlink(final_output_path)
                
        except Exception as e:
            yield f"data: {json.dumps({'progress': 0, 'error': str(e)})}\n\n"
            if os.path.exists(temp_video.name):
                os.unlink(temp_video.name)
    
    return StreamingResponse(
        generate_progress(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )