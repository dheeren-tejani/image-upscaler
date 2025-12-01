# backend/app/services/super_resolution_service.py
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import io
import base64
import os
import numpy as np
import cv2
import time
from app.core.config import settings
from app.services.sr_model_definition import SRModel, SRModel4x, SRModel2xCrazy, SRModel8x # Import all models
from typing import Optional
from collections import OrderedDict

# Try to import TensorRT, but don't fail if it's not available
try:
    import tensorrt as trt
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("SUPER_RES_SERVICE: TensorRT not available. PyTorch models will be used as fallback.")

# --- Global Model Instances and Device ---
# Only load: 2x Normal (TensorRT), 2x Crazy (PyTorch), 4x (TensorRT), 8x (PyTorch)
sr_model_2x_crazy: Optional[SRModel2xCrazy] = None
sr_model_8x: Optional[SRModel8x] = None
trt_infer_2x: Optional['DynamicTensorRTInfer'] = None
trt_infer_4x: Optional['DynamicTensorRTInfer'] = None
device = torch.device(settings.SR_MODEL_DEVICE)

# --- Constants ---
MAX_ENCODER_DOWNSAMPLE_FACTOR = 32 # For ResNet50 up to layer4, this is the factor by which spatial dimensions are reduced.
OPT_INPUT_SHAPE = (1, 3, 1088, 1920)  # Optimal input shape for TensorRT engines

# --- TensorRT Inference Class ---
class DynamicTensorRTInfer:
    """TensorRT inference engine wrapper for batch processing with dynamic shapes."""
    def __init__(self, engine_path: str, batch_size: int):
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT is not available. Cannot initialize TensorRT engine.")
        
        print(f"SUPER_RES_SERVICE: Initializing TensorRT inference engine from {engine_path}")
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        try:
            with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT engine: {e}")
        
        self.context = self.engine.create_execution_context()
        self.input_name = self.engine.get_tensor_name(0)
        self.output_name = self.engine.get_tensor_name(1)
        self.stream = torch.cuda.Stream()
        self.batch_size = batch_size
        
        print(f"SUPER_RES_SERVICE: TensorRT engine loaded successfully (supports dynamic shapes)")
    
    def infer_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        """Runs inference on a batch of tensors with dynamic shapes."""
        # Set input shape dynamically
        self.context.set_input_shape(self.input_name, batch_tensor.shape)
        
        # Get output shape after setting input shape (TensorRT will resolve dynamic dimensions)
        output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        
        # Check if output shape has dynamic dimensions (-1)
        if -1 in output_shape:
            # Calculate expected output shape based on input and model architecture
            # For upscaling models, output is typically scale_factor * input size
            b, c, h, w = batch_tensor.shape
            # Try to infer scale factor from engine or use max possible
            # Most models upscale by 2x or 4x, so estimate conservatively
            estimated_h = h * 4  # Max upscale factor
            estimated_w = w * 4
            # Replace -1 with estimated dimensions
            output_shape = tuple(estimated_h if dim == -1 else dim for dim in output_shape)
            output_shape = tuple(estimated_w if dim == -1 else dim for dim in output_shape)
        
        d_output = torch.empty(output_shape, dtype=torch.float32, device='cuda')
        
        # Use contiguous tensor for input
        d_input = batch_tensor.contiguous()
        
        self.context.set_tensor_address(self.input_name, d_input.data_ptr())
        self.context.set_tensor_address(self.output_name, d_output.data_ptr())
        
        self.context.execute_async_v3(stream_handle=self.stream.cuda_stream)
        self.stream.synchronize()
        
        # Get actual output shape after execution (TensorRT resolves dynamic dims)
        actual_output_shape = tuple(self.context.get_tensor_shape(self.output_name))
        
        # If actual shape is different, we need to handle it
        # For now, if shapes match or actual is smaller, we're good
        if actual_output_shape != output_shape:
            # If actual is smaller, crop; if larger, we allocated enough
            if len(actual_output_shape) == len(output_shape):
                # Compare dimension by dimension
                if actual_output_shape[2] < output_shape[2] or actual_output_shape[3] < output_shape[3]:
                    d_output = d_output[:, :, :actual_output_shape[2], :actual_output_shape[3]]
        
        return d_output

# --- TensorRT Model Loading ---
def load_tensorrt_engine_2x() -> Optional['DynamicTensorRTInfer']:
    """Load TensorRT engine for 2x (Normal) model."""
    global trt_infer_2x
    
    if not TENSORRT_AVAILABLE:
        print("SUPER_RES_SERVICE: TensorRT not available, skipping TensorRT engine load for 2x")
        return None
    
    if trt_infer_2x is not None:
        return trt_infer_2x
    
    if not os.path.exists(settings.SR_2X_TENSORRT_ENGINE_PATH):
        print(f"SUPER_RES_SERVICE: TensorRT engine not found at {settings.SR_2X_TENSORRT_ENGINE_PATH}, using PyTorch model")
        return None
    
    try:
        trt_infer_2x = DynamicTensorRTInfer(settings.SR_2X_TENSORRT_ENGINE_PATH, settings.SR_2X_TENSORRT_BATCH_SIZE)
        print("SUPER_RES_SERVICE: 2x TensorRT engine loaded successfully")
        return trt_infer_2x
    except Exception as e:
        print(f"SUPER_RES_SERVICE ERROR: Could not load 2x TensorRT engine: {e}")
        import traceback
        traceback.print_exc()
        trt_infer_2x = None
        return None

def load_tensorrt_engine_4x() -> Optional['DynamicTensorRTInfer']:
    """Load TensorRT engine for 4x model."""
    global trt_infer_4x
    
    if not TENSORRT_AVAILABLE:
        print("SUPER_RES_SERVICE: TensorRT not available, skipping TensorRT engine load for 4x")
        return None
    
    if trt_infer_4x is not None:
        return trt_infer_4x
    
    if not os.path.exists(settings.SR_4X_TENSORRT_ENGINE_PATH):
        print(f"SUPER_RES_SERVICE ERROR: TensorRT engine not found at {settings.SR_4X_TENSORRT_ENGINE_PATH}")
        return None
    
    try:
        trt_infer_4x = DynamicTensorRTInfer(settings.SR_4X_TENSORRT_ENGINE_PATH, settings.SR_4X_TENSORRT_BATCH_SIZE)
        print("SUPER_RES_SERVICE: 4x TensorRT engine loaded successfully")
        return trt_infer_4x
    except Exception as e:
        print(f"SUPER_RES_SERVICE ERROR: Could not load 4x TensorRT engine: {e}")
        import traceback
        traceback.print_exc()
        trt_infer_4x = None
        return None

# --- TensorRT Inference Functions ---
def run_tensorrt_inference_2x(trt_infer: 'DynamicTensorRTInfer', image_pil: Image.Image) -> Image.Image:
    """Run TensorRT inference for 2x upscaling on a single image (duplicated to batch of 2)."""
    orig_w, orig_h = image_pil.size
    
    # Resize to optimal input shape
    opt_height, opt_width = OPT_INPUT_SHAPE[2], OPT_INPUT_SHAPE[3]
    resized_pil = image_pil.resize((opt_width, opt_height), Image.BILINEAR)
    
    # Convert to tensor and normalize (same as video_enhancer.py)
    image_array = np.array(resized_pil)
    batch_arrays = np.stack([image_array, image_array])
    batch_tensor_bgr = torch.from_numpy(batch_arrays).permute(0, 3, 1, 2).to('cuda')
    batch_tensor_rgb = batch_tensor_bgr.flip(dims=[1])
    batch_tensor_float = batch_tensor_rgb.float() / 255.0
    batch_tensor_normalized = TF.normalize(batch_tensor_float, [0.5]*3, [0.5]*3)
    
    # Run inference
    hr_batch_gpu_large = trt_infer.infer_batch(batch_tensor_normalized)
    
    # Resize to target output size
    target_output_height, target_output_width = orig_h * 2, orig_w * 2
    hr_batch_gpu_resized = F.interpolate(hr_batch_gpu_large, size=(target_output_height, target_output_width), mode='bilinear', align_corners=False)
    
    # Denormalize and convert back
    hr_batch_denormalized = torch.clamp((hr_batch_gpu_resized + 1.0) / 2.0, 0, 1) * 255
    hr_batch_uint8 = hr_batch_denormalized.byte()
    hr_batch_cpu = hr_batch_uint8.permute(0, 2, 3, 1).cpu().numpy()
    
    # Take first result (since we duplicated the image)
    output_frame_rgb = hr_batch_cpu[0]
    output_image_pil = Image.fromarray(output_frame_rgb.astype(np.uint8))
    
    return output_image_pil

def run_tensorrt_inference_4x(trt_infer: 'DynamicTensorRTInfer', image_pil: Image.Image) -> Image.Image:
    """Run TensorRT inference for 4x upscaling on a single image (batch size 1 for dynamic engine)."""
    orig_w, orig_h = image_pil.size
    
    # Convert to numpy array (single image, batch size 1)
    image_array = np.array(image_pil)
    
    # Convert to tensor (same as video-enhancer-4x.py but for single image)
    batch_tensor_bgr = torch.as_tensor(image_array[np.newaxis, ...], device='cuda')  # Add batch dimension
    batch_tensor_rgb = batch_tensor_bgr.permute(0, 3, 1, 2).flip(dims=[1])
    batch_tensor_float_0_1 = batch_tensor_rgb.float() / 255.0
    
    # Resize to optimal input shape if needed (for very large images)
    opt_height, opt_width = OPT_INPUT_SHAPE[2], OPT_INPUT_SHAPE[3]
    _, _, h, w = batch_tensor_float_0_1.shape
    if h > opt_height or w > opt_width:
        # Resize if image is larger than optimal
        batch_tensor_resized = F.interpolate(batch_tensor_float_0_1, size=(opt_height, opt_width), mode='bilinear', align_corners=False)
    else:
        batch_tensor_resized = batch_tensor_float_0_1
    
    # Normalize to -1 to 1 range
    batch_tensor_neg1_1 = (batch_tensor_resized * 2.0) - 1.0
    
    # Run inference
    hr_batch_gpu_large = trt_infer.infer_batch(batch_tensor_neg1_1)
    
    # Resize to target output size
    target_output_height, target_output_width = orig_h * 4, orig_w * 4
    hr_batch_gpu_resized = F.interpolate(hr_batch_gpu_large, size=(target_output_height, target_output_width), mode='bilinear', align_corners=False)
    
    # Denormalize and convert back
    hr_batch_denormalized = torch.clamp((hr_batch_gpu_resized + 1.0) / 2.0, 0, 1) * 255
    hr_batch_uint8 = hr_batch_denormalized.byte()
    hr_batch_cpu = hr_batch_uint8.permute(0, 2, 3, 1).cpu().numpy()
    
    # Get result (batch size 1)
    output_frame_rgb = hr_batch_cpu[0]
    output_image_pil = Image.fromarray(output_frame_rgb.astype(np.uint8))
    
    return output_image_pil

# --- 8x Model Intelligent Patch Size Determination ---
def determine_patch_size_8x(width: int, height: int) -> tuple[int, int, str]:
    """Intelligently determines patch size and overlap based on image resolution for 8x model."""
    total_pixels = width * height
    
    if total_pixels < (1024*768): # Small images
        patch_size = 128
        overlap = 32
        category = "Small"
    elif total_pixels < (1920*1080): # Medium images (HD)
        patch_size = 96
        overlap = 24
        category = "Medium"
    elif total_pixels < (3840*2160): # Large images (2K to 4K)
        patch_size = 64
        overlap = 16
        category = "Large"
    else: # Very large images
        patch_size = 48
        overlap = 12
        category = "X-Large"
        
    print(f"8X_MODEL: Image category: {category} ({width}x{height}). Auto-selected patch_size={patch_size}, overlap={overlap}")
    return patch_size, overlap, category

# --- 8x Model Tiled Inference with Post-Processing ---
def run_tiled_inference_8x(model, lr_image_pil: Image.Image, patch_size: int, overlap: int, device) -> np.ndarray:
    """Run tiled inference for 8x model with sophisticated post-processing."""
    print(f"8X_MODEL: Starting tiled inference with patch_size={patch_size}, overlap={overlap}")
    
    # Convert PIL to tensor
    lr_tensor = transforms.ToTensor()(lr_image_pil).unsqueeze(0).to(device)
    b, c, h, w = lr_tensor.shape
    h_hr, w_hr = h * 8, w * 8
    output_hr = torch.zeros((b, c, h_hr, w_hr), device=device)
    weight_map = torch.zeros_like(output_hr)
    stride = patch_size - overlap
    
    print(f"8X_MODEL: Processing {h}x{w} -> {h_hr}x{w_hr} with {((h-1)//stride + 1) * ((w-1)//stride + 1)} patches")
    
    with torch.no_grad():
        for i in range(0, h, stride):
            for j in range(0, w, stride):
                h_end = min(i + patch_size, h)
                w_end = min(j + patch_size, w)
                lr_patch = lr_tensor[:, :, i:h_end, j:w_end]
                lr_patch_normalized = transforms.Normalize([0.5]*3, [0.5]*3)(lr_patch)
                hr_patch_generated = model(lr_patch_normalized)
                h_start_hr, w_start_hr = i * 8, j * 8
                output_hr[:, :, h_start_hr:h_start_hr+hr_patch_generated.shape[2], w_start_hr:w_start_hr+hr_patch_generated.shape[3]] += hr_patch_generated
                weight_map[:, :, h_start_hr:h_start_hr+hr_patch_generated.shape[2], w_start_hr:w_start_hr+hr_patch_generated.shape[3]] += 1
    
    final_output = output_hr / (weight_map + 1e-8)
    final_output = torch.clamp((final_output.squeeze(0) + 1.0) / 2.0, 0.0, 1.0)
    final_output_numpy = final_output.detach().cpu().permute(1, 2, 0).numpy()
    
    print("8X_MODEL: Applying post-processing: Color Correction and Median Denoising.")
    
    # Post-processing: Color correction and median denoising
    ai_output_bgr = cv2.cvtColor((final_output_numpy * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    bicubic_upscale = lr_image_pil.resize((w_hr, h_hr), Image.BICUBIC)
    bicubic_bgr = cv2.cvtColor(np.array(bicubic_upscale), cv2.COLOR_RGB2BGR)
    ai_output_ycrcb = cv2.cvtColor(ai_output_bgr, cv2.COLOR_BGR2YCrCb)
    bicubic_ycrcb = cv2.cvtColor(bicubic_bgr, cv2.COLOR_BGR2YCrCb)
    ai_y = ai_output_ycrcb[:, :, 0]
    bicubic_cr = bicubic_ycrcb[:, :, 1]
    bicubic_cb = bicubic_ycrcb[:, :, 2]
    color_corrected_ycrcb = np.stack([ai_y, bicubic_cr, bicubic_cb], axis=-1)
    color_corrected_bgr = cv2.cvtColor(color_corrected_ycrcb, cv2.COLOR_YCrCb2BGR)
    color_corrected_rgb = cv2.cvtColor(color_corrected_bgr, cv2.COLOR_BGR2RGB)
    final_polished_image = cv2.medianBlur(color_corrected_rgb, 3)
    
    print("8X_MODEL: Post-processing completed successfully.")
    return final_polished_image

# --- Model Loading ---
def get_sr_model_instance(scale_factor: int = 2, mode: str = "Normal") -> Optional[SRModel2xCrazy | SRModel8x | DynamicTensorRTInfer]:
    """Returns the loaded SR model instance for the specified scale factor and mode.
    For 2x Normal and 4x, returns TensorRT engine. For 2x Crazy and 8x, returns PyTorch model."""
    global sr_model_2x_crazy, sr_model_8x
    
    if scale_factor == 2:
        if mode == "Normal":
            # Use TensorRT for 2x Normal
            return load_tensorrt_engine_2x()
        elif mode == "Crazy":
            if sr_model_2x_crazy is None:
                print("SUPER_RES_SERVICE (get_sr_model_instance): 2x Crazy SR Model is None, attempting to load...")
                return load_sr_model_2x_crazy()
            return sr_model_2x_crazy
        else:
            print(f"SUPER_RES_SERVICE ERROR: Unsupported mode {mode} for 2x upscaling. Only 'Normal' and 'Crazy' are supported.")
            return None
    elif scale_factor == 4:
        # Use TensorRT for 4x
        return load_tensorrt_engine_4x()
    elif scale_factor == 8:
        if sr_model_8x is None:
            print("SUPER_RES_SERVICE (get_sr_model_instance): 8x SR Model is None, attempting to load...")
            return load_sr_model_8x()
        return sr_model_8x
    else:
        print(f"SUPER_RES_SERVICE ERROR: Unsupported scale factor {scale_factor}. Only 2x, 4x, and 8x are supported.")
        return None

# Removed load_sr_model_2x_normal - using TensorRT instead

def load_sr_model_2x_crazy() -> Optional[SRModel2xCrazy]:
    global sr_model_2x_crazy
    
    print(f"SUPER_RES_SERVICE (load_sr_model_2x_crazy): Attempting to load 2x Crazy SR model from {settings.SR_2X_CRAZY_MODEL_PATH} onto {device}...")
    if not os.path.exists(settings.SR_2X_CRAZY_MODEL_PATH):
        print(f"SUPER_RES_SERVICE ERROR: 2x Crazy SR Model checkpoint file not found at {settings.SR_2X_CRAZY_MODEL_PATH}")
        sr_model_2x_crazy = None
        return None

    try:
        model_instance = SRModel2xCrazy()
        checkpoint = torch.load(settings.SR_2X_CRAZY_MODEL_PATH, map_location=device)
        
        state_dict_to_load = None
        epoch_trained = 'N/A'
        best_val_loss_str = 'N/A'

        if 'generator_state_dict' in checkpoint and isinstance(checkpoint['generator_state_dict'], OrderedDict):
            state_dict_to_load = checkpoint['generator_state_dict']
            epoch_trained = checkpoint.get('epoch', 'N/A')
            best_val_loss = checkpoint.get('generator_loss')
            if isinstance(best_val_loss, float):
                best_val_loss_str = f"{best_val_loss:.6f}"
        elif isinstance(checkpoint, OrderedDict):
            state_dict_to_load = checkpoint
            epoch_trained = 'N/A (direct state_dict loaded)'
        else:
            print(f"SUPER_RES_SERVICE ERROR: 2x Crazy Checkpoint is not in a recognized format.")
            sr_model_2x_crazy = None
            return None

        new_state_dict = OrderedDict()
        for k, v in state_dict_to_load.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model_instance.load_state_dict(new_state_dict)
        model_instance.to(device)
        model_instance.eval()
        sr_model_2x_crazy = model_instance
        print(f"SUPER_RES_SERVICE: 2x Crazy SR Model loaded successfully. Trained epoch: {epoch_trained}, Best Val Loss: {best_val_loss_str}")
        return sr_model_2x_crazy
    except Exception as e:
        print(f"SUPER_RES_SERVICE ERROR: Could not load 2x Crazy SR model: {e}")
        import traceback
        traceback.print_exc()
        sr_model_2x_crazy = None
        return None

# Removed load_sr_model_4x - using TensorRT instead

def load_sr_model_8x() -> Optional[SRModel8x]:
    global sr_model_8x
    
    print(f"SUPER_RES_SERVICE (load_sr_model_8x): Attempting to load 8x SR model from {settings.SR_8X_MODEL_PATH} onto {device}...")
    if not os.path.exists(settings.SR_8X_MODEL_PATH):
        print(f"SUPER_RES_SERVICE ERROR: 8x SR Model checkpoint file not found at {settings.SR_8X_MODEL_PATH}")
        sr_model_8x = None
        return None

    try:
        model_instance = SRModel8x(upscale_factor=settings.SR_8X_UPSCALE_FACTOR, encoder_pretrained=False)
        checkpoint = torch.load(settings.SR_8X_MODEL_PATH, map_location=device)
        
        state_dict_to_load = None
        epoch_trained = 'N/A'
        best_val_loss_str = 'N/A'

        if 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], OrderedDict):
            state_dict_to_load = checkpoint['model_state_dict']
            epoch_trained = checkpoint.get('epoch', 'N/A')
            best_val_loss = checkpoint.get('best_val_loss')
            if isinstance(best_val_loss, float):
                best_val_loss_str = f"{best_val_loss:.6f}"
        elif isinstance(checkpoint, OrderedDict):
            state_dict_to_load = checkpoint
            epoch_trained = 'N/A (direct state_dict loaded)'
        else:
            print(f"SUPER_RES_SERVICE ERROR: 8x Checkpoint is not in a recognized format.")
            sr_model_8x = None
            return None

        new_state_dict = OrderedDict()
        for k, v in state_dict_to_load.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        
        model_instance.load_state_dict(new_state_dict)
        model_instance.to(device)
        model_instance.eval()
        sr_model_8x = model_instance
        print(f"SUPER_RES_SERVICE: 8x SR Model loaded successfully. Trained epoch: {epoch_trained}, Best Val Loss: {best_val_loss_str}")
        return sr_model_8x
    except Exception as e:
        print(f"SUPER_RES_SERVICE ERROR: Could not load 8x SR model: {e}")
        import traceback
        traceback.print_exc()
        sr_model_8x = None
        return None

# --- Image Preprocessing and Postprocessing ---
preprocess_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def tensor_to_pil(tensor_image: torch.Tensor) -> Image.Image:
    image = tensor_image.clone().detach().squeeze(0).cpu().permute(1, 2, 0) 
    image = (image * 0.5) + 0.5
    image_np = image.numpy().clip(0, 1)
    return Image.fromarray((image_np * 255).astype(np.uint8))

# --- Main Upscaling Function ---
async def upscale_image_bytes(image_bytes: bytes, scale_factor: int = 2, mode: str = "Normal") -> Optional[str]:
    try:
        input_image_pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        print(f"SUPER_RES_SERVICE ERROR: Could not open input image bytes: {e}")
        return None 

    orig_w, orig_h = input_image_pil.size
    print(f"SUPER_RES_SERVICE: Original LR image size: {orig_w}x{orig_h}")
    
    # Start timing
    start_time = time.time()
    print(f"SUPER_RES_SERVICE: Starting {scale_factor}x {mode} upscaling...")
    
    target_hr_w = orig_w * scale_factor
    target_hr_h = orig_h * scale_factor

    # Get model instance (TensorRT for 2x Normal and 4x, PyTorch for others)
    current_sr_model = get_sr_model_instance(scale_factor, mode)
    
    if current_sr_model is None:
        print(f"SUPER_RES_SERVICE (upscale_image_bytes): {scale_factor}x {mode} SR Model not available for upscaling.")
        return "MODEL_NOT_LOADED_ERROR"
    
    # Check if it's a TensorRT engine
    is_tensorrt = isinstance(current_sr_model, DynamicTensorRTInfer)
    
    if is_tensorrt:
        # Use TensorRT inference (2x Normal or 4x)
        try:
            if scale_factor == 2:
                output_image_pil = run_tensorrt_inference_2x(current_sr_model, input_image_pil)
            elif scale_factor == 4:
                output_image_pil = run_tensorrt_inference_4x(current_sr_model, input_image_pil)
            else:
                raise ValueError(f"TensorRT not supported for scale_factor={scale_factor}")
            
            # End timing
            end_time = time.time()
            processing_time = end_time - start_time
            print(f"SUPER_RES_SERVICE: {scale_factor}x {mode} TensorRT upscaling completed in {processing_time:.2f} seconds")
            print(f"SUPER_RES_SERVICE: Image upscaled successfully to {output_image_pil.width}x{output_image_pil.height}.")
            
            buffered = io.BytesIO()
            output_image_pil.save(buffered, format="PNG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return base64_image
            
        except Exception as e:
            print(f"SUPER_RES_SERVICE ERROR: TensorRT inference failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        # Use PyTorch model (for 2x Crazy and 8x)
        if scale_factor == 8:
            # Special handling for 8x model with tiled inference
            try:
                # Determine optimal patch size
                patch_size, overlap, category = determine_patch_size_8x(orig_w, orig_h)
                
                # Run tiled inference with post-processing
                final_output_numpy = run_tiled_inference_8x(current_sr_model, input_image_pil, patch_size, overlap, device)
                
                # Convert numpy array to PIL
                output_image_pil = Image.fromarray(final_output_numpy)
            
            except Exception as e:
                print(f"SUPER_RES_SERVICE ERROR: 8x tiled inference failed: {e}")
                import traceback
                traceback.print_exc()
                return None
        else:
            # Standard processing for 2x (Crazy) and any PyTorch 4x fallback
            # Calculate padding needed for LR image to be divisible by the model's max downsample factor
            pad_w = 0
            if orig_w % MAX_ENCODER_DOWNSAMPLE_FACTOR != 0:
                pad_w = MAX_ENCODER_DOWNSAMPLE_FACTOR - (orig_w % MAX_ENCODER_DOWNSAMPLE_FACTOR)
            
            pad_h = 0
            if orig_h % MAX_ENCODER_DOWNSAMPLE_FACTOR != 0:
                pad_h = MAX_ENCODER_DOWNSAMPLE_FACTOR - (orig_h % MAX_ENCODER_DOWNSAMPLE_FACTOR)

            input_image_for_model = input_image_pil
            
            if pad_w > 0 or pad_h > 0:
                # Pad only right and bottom: (left, top, right_pad, bottom_pad)
                padding_values = (0, 0, pad_w, pad_h) 
                padding_transform = transforms.Pad(padding_values, fill=0, padding_mode='reflect')
                input_image_for_model = padding_transform(input_image_pil)
                print(f"SUPER_RES_SERVICE: Input LR image ({orig_w}x{orig_h}) padded to ({input_image_for_model.width}x{input_image_for_model.height}) for model compatibility.")
            else:
                print(f"SUPER_RES_SERVICE: Input LR image ({orig_w}x{orig_h}) requires no padding.")

            input_tensor = preprocess_transform(input_image_for_model).unsqueeze(0).to(device)
            print(f"SUPER_RES_SERVICE: Model input tensor shape: {input_tensor.shape}")

            with torch.no_grad():
                try:
                    if str(device) == "cuda" and torch.cuda.is_available():
                        with torch.amp.autocast(device_type="cuda", enabled=True):
                            output_hr_padded_tensor = current_sr_model(input_tensor)
                    else:
                        output_hr_padded_tensor = current_sr_model(input_tensor)
                    print(f"SUPER_RES_SERVICE: Model output (padded HR) tensor shape: {output_hr_padded_tensor.shape}")
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and str(device) == "cuda":
                        print(f"SUPER_RES_SERVICE: CUDA out of memory during SR inference. Input LR size: {orig_w}x{orig_h}. Padded LR size: {input_image_for_model.width}x{input_image_for_model.height}. Try smaller image.")
                        if torch.cuda.is_available(): torch.cuda.empty_cache()
                        return "CUDA_OOM_ERROR"
                    print(f"SUPER_RES_SERVICE: Runtime error during model inference: {e}")
                    import traceback
                    traceback.print_exc()
                    raise e 
                except Exception as e_inf:
                    print(f"SUPER_RES_SERVICE: Unexpected error during model inference: {e_inf}")
                    import traceback
                    traceback.print_exc()
                    raise e_inf

            # Crop the output tensor back to the target HR size (corresponding to original LR * upscale_factor)
            # Since we padded only right and bottom, we just need to take the top-left portion.
            final_output_tensor = output_hr_padded_tensor[:, :, :target_hr_h, :target_hr_w]
            print(f"SUPER_RES_SERVICE: Cropped final HR tensor shape: {final_output_tensor.shape} (Target: {target_hr_h}x{target_hr_w})")
                    
            output_image_pil = tensor_to_pil(final_output_tensor)
    
    # End timing
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"SUPER_RES_SERVICE: {scale_factor}x {mode} upscaling completed in {processing_time:.2f} seconds")
    print(f"SUPER_RES_SERVICE: Image upscaled successfully to {output_image_pil.width}x{output_image_pil.height}.")
    
    buffered = io.BytesIO()
    output_image_pil.save(buffered, format="PNG") # Consider "JPEG" for smaller size, or "WEBP"
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return base64_image

# --- Initial Model Load Attempt ---
# This line will be executed when the module is first imported (e.g., on FastAPI startup)
# It ensures an attempt is made to load the model early.
if __name__ != "__main__": # Standard check to run only when imported, not when script is run directly
    # Load only required models: 2x Normal (TensorRT), 2x Crazy (PyTorch), 4x (TensorRT), 8x (PyTorch)
    print("SUPER_RES_SERVICE: Loading required models...")
    
    # Load TensorRT engines
    if TENSORRT_AVAILABLE:
        print("SUPER_RES_SERVICE: Loading TensorRT engines...")
        load_tensorrt_engine_2x()
        load_tensorrt_engine_4x()
    
    # Load PyTorch models
    if sr_model_2x_crazy is None:
        print("SUPER_RES_SERVICE: Loading 2x Crazy model...")
        load_sr_model_2x_crazy()
    if sr_model_8x is None:
        print("SUPER_RES_SERVICE: Loading 8x model...")
        load_sr_model_8x()