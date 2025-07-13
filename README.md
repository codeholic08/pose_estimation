# Face Landmark Detection

This repository contains two approaches for detecting face landmarks and adding landmark dots to images:

1. **ONNX-based approach** - Uses your local ONNX model directly
2. **MediaPipe API approach** - Uses MediaPipe's Python API (recommended for beginners)

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Method 1: ONNX Model Approach

Use this method if you want to work directly with your ONNX model file.

### Usage

```bash
python face_landmark_detector.py --image A1.jpg --model models/mediapipe_face.onnx
```

### Options

- `--image, -i`: Path to input image (required)
- `--model, -m`: Path to ONNX model file (required)
- `--output, -o`: Path to output image (optional)
- `--dot-size`: Size of landmark dots (default: 2)
- `--dot-color`: Color of dots in BGR format (default: [0, 255, 0] - green)
- `--show-numbers`: Show landmark numbers on the image

### Examples

```bash
# Basic usage
python face_landmark_detector.py --image A1.jpg --model models/mediapipe_face.onnx

# Custom output file and red dots
python face_landmark_detector.py --image A1.jpg --model models/mediapipe_face.onnx --output result.jpg --dot-color 0 0 255

# Larger dots with numbers
python face_landmark_detector.py --image A1.jpg --model models/mediapipe_face.onnx --dot-size 4 --show-numbers
```

## Method 2: MediaPipe API Approach (Recommended)

This method uses MediaPipe's Python API and is generally more reliable and easier to use.

### Usage

```bash
python mediapipe_face_landmarks.py --image A1.jpg
```

### Options

- `--image, -i`: Path to input image (required)
- `--output, -o`: Path to output image (optional)
- `--dot-size`: Size of landmark dots (default: 1)
- `--dot-color`: Color of dots in BGR format (default: [0, 255, 0] - green)
- `--style`: Drawing style - 'dots', 'mesh', or 'both' (default: 'dots')
- `--max-faces`: Maximum number of faces to detect (default: 1)
- `--confidence`: Minimum detection confidence (default: 0.5)

### Examples

```bash
# Basic usage - just dots
python mediapipe_face_landmarks.py --image A1.jpg

# Show full face mesh
python mediapipe_face_landmarks.py --image A1.jpg --style mesh

# Both dots and mesh with custom colors
python mediapipe_face_landmarks.py --image A1.jpg --style both --dot-color 255 0 0

# Detect multiple faces with lower confidence
python mediapipe_face_landmarks.py --image A1.jpg --max-faces 5 --confidence 0.3

# Larger dots for better visibility
python mediapipe_face_landmarks.py --image A1.jpg --dot-size 3 --dot-color 0 0 255
```

## Output

Both scripts will:
1. Display the result image in a window
2. Save the result image with landmarks to disk
3. Print detection information to the console

The output filename will be automatically generated based on the input filename:
- ONNX approach: `A1_landmarks.jpg`
- MediaPipe approach: `A1_mediapipe_landmarks.jpg`

## Features

### ONNX Approach Features:
- Direct ONNX model inference
- Automatic model input/output handling
- Fallback face detection using OpenCV
- Customizable dot visualization
- Support for different model formats

### MediaPipe Approach Features:
- 468 facial landmarks detection
- Face mesh visualization with connections
- Iris landmark detection (if available)
- Multiple face detection
- Adjustable confidence thresholds
- Three visualization styles (dots, mesh, both)

## Troubleshooting

### Common Issues

1. **"No landmarks detected"**
   - Try lowering the confidence threshold: `--confidence 0.3`
   - Ensure the face is clearly visible and well-lit
   - Check if the image contains a face

2. **"Model file not found"**
   - Verify the path to your ONNX model file
   - Ensure the model file is in the correct location

3. **"Image file not found"**
   - Check the image file path
   - Ensure the image file exists and is readable

4. **Poor landmark detection**
   - Try different confidence thresholds
   - Ensure good lighting in the image
   - Use higher resolution images

### Performance Tips

- For better performance with the ONNX approach, use GPU acceleration if available
- MediaPipe approach is generally faster and more accurate
- Use `static_image_mode=True` for single images (default in our script)

## Dependencies

- **opencv-python**: Image processing and visualization
- **numpy**: Numerical operations
- **onnxruntime**: ONNX model inference (for ONNX approach)
- **mediapipe**: Face landmark detection (for MediaPipe approach)
- **Pillow**: Image handling utilities

## File Structure

```
pose_estimation/
├── face_landmark_detector.py      # ONNX-based approach
├── mediapipe_face_landmarks.py    # MediaPipe API approach
├── requirements.txt               # Dependencies
├── README.md                     # This file
├── A1.jpg                        # Your input image
└── models/
    └── mediapipe_face.onnx       # Your ONNX model
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run with MediaPipe (recommended):
   ```bash
   python mediapipe_face_landmarks.py --image A1.jpg
   ```

3. Or run with your ONNX model:
   ```bash
   python face_landmark_detector.py --image A1.jpg --model models/mediapipe_face.onnx
   ```

The result will be displayed and saved automatically! 