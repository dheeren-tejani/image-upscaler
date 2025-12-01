# backend/app/core/config.py
from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
import torch # For device detection

load_dotenv()

class Settings(BaseSettings):
    APP_NAME: str = "Super Resolution API"
    
    # Determine project root to make model path relative
    # This assumes 'config.py' is in 'app/core/'
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # 2x Model Configuration (Normal)
    SR_MODEL_FILENAME: str = "final_sr_model_weights.pth"
    SR_MODEL_DIR: str = os.path.join(PROJECT_ROOT, "ml_models", "sr_model", "Model_1")
    SR_MODEL_PATH: str = os.path.join(SR_MODEL_DIR, SR_MODEL_FILENAME)
    SR_UPSCALE_FACTOR: int = 2 # Must match your trained model
    # TensorRT Engine for 2x (Normal) - batch size 2
    SR_2X_TENSORRT_ENGINE_FILENAME: str = "model_2x_batch_2_int8_trial_2.engine"
    SR_2X_TENSORRT_ENGINE_PATH: str = os.path.join(SR_MODEL_DIR, SR_2X_TENSORRT_ENGINE_FILENAME)
    SR_2X_TENSORRT_BATCH_SIZE: int = 2
    
    # 2x Model Configuration (Crazy)
    SR_2X_CRAZY_MODEL_FILENAME: str = "latest_gan_checkpoint.pth"
    SR_2X_CRAZY_MODEL_DIR: str = os.path.join(PROJECT_ROOT, "ml_models", "sr_model", "Model_2")
    SR_2X_CRAZY_MODEL_PATH: str = os.path.join(SR_2X_CRAZY_MODEL_DIR, SR_2X_CRAZY_MODEL_FILENAME)
    SR_2X_CRAZY_UPSCALE_FACTOR: int = 2
    
    # 4x Model Configuration
    # Keep original PyTorch checkpoint path defined for compatibility/logging,
    # even though runtime uses the TensorRT engine for inference.
    SR_4X_MODEL_FILENAME: str = "4x_upscaler.pth"
    SR_4X_MODEL_DIR: str = os.path.join(PROJECT_ROOT, "ml_models", "sr_model", "Model_3_new")
    SR_4X_MODEL_PATH: str = os.path.join(SR_4X_MODEL_DIR, SR_4X_MODEL_FILENAME)
    SR_4X_UPSCALE_FACTOR: int = 4
    # TensorRT Engine for 4x - engine built with batch size 1 and dynamic shapes
    SR_4X_TENSORRT_ENGINE_DIR: str = os.path.join(PROJECT_ROOT, "ml_models", "sr_model", "Model_4")
    SR_4X_TENSORRT_ENGINE_FILENAME: str = "model_4x_batch_2_int8.engine"
    SR_4X_TENSORRT_ENGINE_PATH: str = os.path.join(SR_4X_TENSORRT_ENGINE_DIR, SR_4X_TENSORRT_ENGINE_FILENAME)
    SR_4X_TENSORRT_BATCH_SIZE: int = 1  # Engine expects batch size 1 with dynamic height/width
    
    # 8x Model Configuration
    SR_8X_MODEL_FILENAME: str = "final_8x_model.pth"
    SR_8X_MODEL_DIR: str = os.path.join(PROJECT_ROOT, "ml_models", "sr_model", "Model_5")
    SR_8X_MODEL_PATH: str = os.path.join(SR_8X_MODEL_DIR, SR_8X_MODEL_FILENAME)
    SR_8X_UPSCALE_FACTOR: int = 8
    
    SR_MODEL_DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

settings = Settings()

print(f"SR 2x Normal Model Path from config: {settings.SR_MODEL_PATH}")
print(f"SR 2x Crazy Model Path from config: {settings.SR_2X_CRAZY_MODEL_PATH}")
print(f"SR 4x Model Path from config: {settings.SR_4X_MODEL_PATH}")
print(f"SR 8x Model Path from config: {settings.SR_8X_MODEL_PATH}")
print(f"SR Model Device from config: {settings.SR_MODEL_DEVICE}")
print(f"SR 2x TensorRT Engine Path from config: {settings.SR_2X_TENSORRT_ENGINE_PATH}")
print(f"SR 4x TensorRT Engine Path from config: {settings.SR_4X_TENSORRT_ENGINE_PATH}")