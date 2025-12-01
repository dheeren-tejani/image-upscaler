# 🎨 AI Image & Video Upscaler

A high-performance, GPU-accelerated image and video upscaling application that brings professional-quality super-resolution to consumer hardware. Built with PyTorch, FastAPI, and React.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7.1-red)
![React](https://img.shields.io/badge/React-18.3-blue)

---

## 🚀 Project Overview

This project democratizes AI-powered image upscaling by making state-of-the-art super-resolution models accessible to users with modest hardware. All models were trained on **Google Colab's free-tier GPUs**, proving that you don't need expensive infrastructure to create powerful AI tools.

### ✨ Key Features

- **Multiple Upscaling Factors**: 2x, 4x, and 8x super-resolution models
- **Dual Modes for 2x**: 
  - **Normal Mode**: Balanced quality and speed
  - **Crazy Mode**: Maximum detail enhancement with GAN-based architecture
- **Video Upscaling**: Real-time progress tracking with Server-Sent Events (SSE)
- **TensorRT Acceleration**: Optimized inference for 2x and 4x models using INT8 quantization
- **Smart Tiled Processing**: Handles large images with the 8x model using intelligent patch-based inference
- **Modern UI**: Sleek, dark-themed interface with image comparison slider and zoom/pan functionality
- **Session Management**: Track and manage multiple upscaling sessions

---

## 🎯 Architecture Highlights

### Backend (FastAPI + PyTorch)
- **Custom ResNet50-based U-Net** architecture with skip connections
- **TensorRT integration** for 2-3x faster inference on NVIDIA GPUs
- **Batch processing** with producer-consumer pattern for video upscaling
- **Smart memory management** with automatic GPU cache clearing
- **Tiled inference** for 8x model to handle arbitrarily large images

### Frontend (React + TypeScript)
- **Real-time progress tracking** via SSE for video processing
- **Interactive image comparison** with draggable slider
- **Zoom & pan** functionality for detailed inspection
- **Responsive design** with Tailwind CSS
- **Session history** with thumbnail previews

---

## 📊 Models & Training

All models were trained using the following datasets:
- **FLICKR2K**: 2,650 high-quality images
- **DIV2K**: 800 diverse 2K resolution images

### Model Specifications

| Model | Architecture | Parameters | Input | Output | Training Platform |
|-------|-------------|------------|-------|--------|------------------|
| 2x Normal | ResNet50 U-Net | ~25M | 256x256 | 512x512 | Google Colab (T4) |
| 2x Crazy | ResNet50 U-Net + GAN | ~28M | 256x256 | 512x512 | Google Colab (T4) |
| 4x | ResNet50 U-Net + GAN | ~25M | 256x256 | 1024x1024 | Google Colab (T4) |
| 8x | Enhanced ResNet50 U-Net | ~30M | Variable | 8x Output | Google Colab (T4) |

**Note**: The 2x Normal and 4x models are served via TensorRT engines for optimal performance.

---

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Node.js 18+
- CUDA-capable GPU (recommended) or CPU
- FFmpeg (for video processing)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/dheeren-tejani/image-upscaler.git
cd ai-upscaler/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download model weights (place in backend/ml_models/)
# Structure:
# backend/ml_models/
# ├── sr_model/
# │   ├── Model_1/  (2x Normal - PyTorch checkpoint + TensorRT engine)
# │   ├── Model_2/  (2x Crazy - PyTorch checkpoint)
# │   ├── Model_4/  (4x - TensorRT engine)
# │   └── Model_5/  (8x - PyTorch checkpoint)

# Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Setup

```bash
cd ../frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The application will be available at `http://localhost:8080`

---

## 💻 Usage

### Image Upscaling

1. **Upload an Image**: Drag & drop or click to select
2. **Choose Settings**: 
   - Select upscale factor (2x, 4x, or 8x)
   - For 2x: Choose between Normal or Crazy mode
3. **Enhance**: Click the "Enhance" button
4. **Compare**: Use the slider to compare original vs upscaled
5. **Download**: Save your enhanced image

### Video Upscaling

1. Switch to **Video Mode** in the control panel
2. Upload a video file (MP4, AVI, MOV, MKV, WebM)
3. Select upscale factor (2x or 4x)
4. Click "Upscale Video"
5. Monitor real-time progress
6. Download the enhanced video

### API Endpoints

```bash
# Image upscaling
POST /api/v1/sr/upscale-image
Content-Type: multipart/form-data
Body: 
  - file: image file
  - scale_factor: 2 | 4 | 8
  - mode: "Normal" | "Crazy" (for 2x only)

# Video upscaling (Server-Sent Events)
POST /api/v1/sr/upscale-video
Content-Type: multipart/form-data
Body:
  - file: video file
  - scale_factor: 2 | 4
  - mode: "Normal" | "Crazy" (for 2x only)
```

---

## 🔧 Technical Details

### TensorRT Optimization

The 2x Normal and 4x models are optimized using TensorRT with INT8 quantization:
- **2x Normal**: Batch size 2, dynamic shapes
- **4x**: Batch size 1, dynamic height/width

This provides **2-3x speedup** over standard PyTorch inference with minimal quality loss.

### Tiled Inference (8x Model)

For very large images, the 8x model uses intelligent tiling:
- Automatically adjusts patch size based on image resolution
- Overlapping patches with weighted blending
- Post-processing with color correction and median denoising

### Video Processing Pipeline

```
Input Video → Frame Extraction → Batch Processing (TensorRT)
   → Frame Assembly → Audio Extraction → Final Encoding (H.264)
```

Uses producer-consumer pattern with queue-based frame buffering for optimal memory usage.

---

## 📈 Performance

Tested on NVIDIA RTX 3060 (12GB VRAM):

| Input Size | Model | Processing Time | Memory Usage |
|-----------|-------|----------------|--------------|
| 512x512 | 2x Normal (TensorRT) | ~0.3s | ~2GB |
| 512x512 | 2x Crazy (PyTorch) | ~0.5s | ~2.5GB |
| 512x512 | 4x (TensorRT) | ~0.8s | ~3GB |
| 1024x1024 | 8x (Tiled) | ~15s | ~4GB |
| 1920x1080 video (30s) | 2x | ~45s | ~3GB |

---

## 🎓 Training Details

### Loss Functions
- **2x Normal**: MSE + Perceptual Loss (VGG16)
- **2x Crazy**: Adversarial Loss + Perceptual Loss + MSE
- **4x**: MSE + SSIM + Perceptual Loss
- **8x**: L1 + Perceptual Loss + Gradient Loss

### Augmentations
- Random horizontal/vertical flips
- Random rotations (90°, 180°, 270°)
- Random crops
- Color jittering

### Training Infrastructure
- Platform: Google Colab (Tesla T4 GPU)
- Training time: ~8-12 hours per model
- Batch size: 16-32 depending on model
- Optimizer: Adam (lr=1e-4)

---

## 🗺️ Future Work

This is an ongoing project with several planned improvements:

- [ ] **Web deployment** on cloud platforms (AWS/GCP)
- [ ] **Model quantization** for mobile devices
- [ ] **Real-time video upscaling** for webcam feeds
- [ ] **Face enhancement** specialized model
- [ ] **Anime/cartoon** specialized models
- [ ] **API rate limiting** and user authentication
- [ ] **Batch processing** for multiple images
- [ ] **Model fine-tuning** on custom datasets

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👨‍💻 Author

**Dheeren Tejani**

Created with the goal of making professional-quality image upscaling accessible to everyone, regardless of their hardware. All models were trained on free-tier Google Colab GPUs to prove that amazing AI tools can be built without expensive infrastructure.

---

## 🙏 Acknowledgments

- **Datasets**: FLICKR2K and DIV2K teams
- **Frameworks**: PyTorch, FastAPI, React, Tailwind CSS
- **Training Platform**: Google Colab
- **Optimization**: NVIDIA TensorRT
- **Inspiration**: Various open-source super-resolution projects

---

## 📧 Contact

For questions, suggestions, or collaborations, feel free to reach out!

---

**⭐ If you find this project useful, please consider giving it a star!**