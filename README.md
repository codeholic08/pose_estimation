# MediaPipe Face Landmark Detection

This repository contains a production-ready face landmark detection system using MediaPipe ONNX models.

## Features

- **468 facial landmarks** detection using MediaPipe ONNX model
- **Accurate face alignment** with automatic face detection and cropping
- **Multiple visualization options** (landmark dots, full face mesh)
- **Production-ready code** with proper error handling
- **Easy-to-use command-line interface**

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Landmark Detection

```bash
python face_landmark_detector_final.py --image your_image.jpg
```

### Face Mesh Visualization

```bash
python face_landmark_detector_final.py --image your_image.jpg --mesh
```

### Advanced Options

```bash
# Custom dot size and colors
python face_landmark_detector_final.py --image your_image.jpg --dot-size 4 --dot-color 255 0 0

# Face mesh with custom line thickness
python face_landmark_detector_final.py --image your_image.jpg --mesh --line-thickness 2

# Verbose output
python face_landmark_detector_final.py --image your_image.jpg --verbose

# No display window (save only)
python face_landmark_detector_final.py --image your_image.jpg --no-display

# Use custom model path (if needed)
python face_landmark_detector_final.py --image your_image.jpg --model path/to/your/model.onnx
```

### Programmatic Usage

You can also use the detection function directly in your Python code:

```python
from face_landmark_detector_final import detect_face_landmarks

# Basic usage
output_path = detect_face_landmarks("your_image.jpg")

# With face mesh
output_path = detect_face_landmarks("your_image.jpg", draw_mesh=True)

# With custom parameters
output_path = detect_face_landmarks(
    "your_image.jpg", 
    draw_mesh=True,
    dot_size=3,
    dot_color=(255, 0, 0),  # Red dots
    line_color=(0, 255, 0), # Green lines
    verbose=True
)
```

## Command Line Arguments

- `--image, -i`: Path to input image (required)
- `--model, -m`: Path to ONNX model file (default: models/mediapipe_face.onnx)
- `--output, -o`: Path to output image (optional, auto-generated if not provided)
- `--dot-size`: Size of landmark dots (default: 2)
- `--dot-color`: Color of dots in BGR format (default: [0, 255, 0])
- `--mesh`: Draw face mesh connections
- `--line-color`: Color of mesh lines in BGR format (default: [255, 0, 0])
- `--line-thickness`: Thickness of mesh lines (default: 1)
- `--verbose, -v`: Verbose output
- `--no-display`: Don't display result window

## Technical Details

- **Model Input**: 192x192 RGB images (automatically cropped from detected face region)
- **Model Output**: 468 facial landmarks with (x, y, z) coordinates
- **Face Detection**: Uses OpenCV's Haar cascade for initial face detection
- **Coordinate System**: Landmarks are properly transformed from face crop space to original image coordinates
- **Performance**: Fast inference with ONNX runtime

## File Structure

```
pose_estimation/
├── face_landmark_detector_final.py  # Main production script
├── example_usage.py                 # Example usage demonstrations
├── requirements.txt                  # Dependencies
├── README.md                        # This file
├── A1.jpg                           # Sample input image
└── models/
    └── mediapipe_face.onnx          # MediaPipe face landmark model
```

## Requirements

- Python 3.7+
- OpenCV
- ONNX Runtime
- NumPy
- Pillow

## License

This project uses MediaPipe models. Please refer to the original MediaPipe licensing terms.

## Notes

- The system automatically detects faces and crops them before landmark detection for optimal accuracy
- Landmarks are properly aligned with facial features (eyes, nose, mouth, face contour)
- The face mesh connections follow the official MediaPipe specification
- Output images are saved in the same directory as the input image by default 