# backend/app/schemas/super_resolution.py
from pydantic import BaseModel
from typing import Optional

class SRResponse(BaseModel):
    upscaled_image_base64: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None # General message, e.g., "Image upscaled successfully"