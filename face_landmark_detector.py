#!/usr/bin/env python3
"""
Face Landmark Detection with ONNX Model
========================================

This script performs face landmark detection using an ONNX model and adds
landmark dots to the input image. It's designed to work with MediaPipe face
models and provides visualization of detected facial landmarks.

Requirements:
- onnxruntime
- opencv-python
- numpy
- Pillow (PIL)

Usage:
    python face_landmark_detector.py --image A1.jpg --model models/mediapipe_face.onnx
"""

import argparse
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
import sys
from typing import List, Tuple, Optional


class FaceLandmarkDetector:
    """Face landmark detector using ONNX model."""
    
    def __init__(self, model_path: str):
        """
        Initialize the face landmark detector.
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_names = None
        self.input_shape = None
        
        self._load_model()
    
    def _load_model(self):
        """Load the ONNX model and get input/output information."""
        try:
            # Create ONNX Runtime session
            providers = ['CPUExecutionProvider']
            if ort.get_device() == 'GPU':
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input information
            input_info = self.session.get_inputs()[0]
            self.input_name = input_info.name
            self.input_shape = input_info.shape
            
            # Get output information
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"Model loaded successfully!")
            print(f"Input name: {self.input_name}")
            print(f"Input shape: {self.input_shape}")
            print(f"Output names: {self.output_names}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess the input image for the model.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Preprocessed image ready for model inference
        """
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get target size from model input shape
        if len(self.input_shape) == 4:  # NCHW format
            target_height = self.input_shape[2]
            target_width = self.input_shape[3]
        else:  # NHWC format
            target_height = self.input_shape[1]
            target_width = self.input_shape[2]
        
        # Resize image
        resized_image = cv2.resize(rgb_image, (target_width, target_height))
        
        # Normalize to [0, 1] range
        normalized_image = resized_image.astype(np.float32) / 255.0
        
        # Add batch dimension and rearrange if needed
        if len(self.input_shape) == 4 and self.input_shape[1] == 3:  # NCHW format
            processed_image = np.transpose(normalized_image, (2, 0, 1))
            processed_image = np.expand_dims(processed_image, axis=0)
        else:  # NHWC format
            processed_image = np.expand_dims(normalized_image, axis=0)
        
        return processed_image
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect face landmarks in the image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Detected landmarks as numpy array or None if no face detected
        """
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: processed_image})
            
            # Process outputs - landmarks should be in the second output (index 1)
            # The model has outputs ['scores', 'landmarks'] so landmarks is at index 1
            landmarks = outputs[1] if len(outputs) > 1 else outputs[0]
            
            # Handle different output formats
            if landmarks.ndim == 3:  # Batch dimension
                landmarks = landmarks[0]
            
            return landmarks
            
        except Exception as e:
            print(f"Error during inference: {e}")
            return None
    
    def postprocess_landmarks(self, landmarks: np.ndarray, 
                            original_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Convert normalized landmarks to pixel coordinates.
        
        Args:
            landmarks: Normalized landmarks from model
            original_shape: Original image shape (height, width)
            
        Returns:
            List of (x, y) pixel coordinates
        """
        height, width = original_shape[:2]
        points = []
        
        # Debug: print landmark shape and some values
        print(f"Landmarks shape: {landmarks.shape}")
        print(f"Landmarks dtype: {landmarks.dtype}")
        print(f"Landmarks min/max: {landmarks.min():.4f}/{landmarks.max():.4f}")
        print(f"Original image size: {width}x{height}")
        
        # MediaPipe face landmarks are in format (N, 3) where N is number of landmarks
        # The coordinates are normalized to [0,1] relative to the input image size (192x192)
        # We need to scale them to the original image size
        if landmarks.ndim == 2 and landmarks.shape[1] == 3:  # Shape: (N, 3)
            for landmark in landmarks:
                # The landmarks are normalized coordinates [0,1] relative to the 192x192 input
                # Scale them to the original image size
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                
                # Clamp to image boundaries
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                
                points.append((x, y))
        elif landmarks.ndim == 2 and landmarks.shape[1] == 2:  # Shape: (N, 2)
            for landmark in landmarks:
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                
                # Clamp to image boundaries
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                
                points.append((x, y))
        elif landmarks.ndim == 1:  # Flattened array
            # Assume alternating x, y coordinates
            for i in range(0, len(landmarks), 2):
                if i + 1 < len(landmarks):
                    x = int(landmarks[i] * width)
                    y = int(landmarks[i + 1] * height)
                    
                    # Clamp to image boundaries
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    
                    points.append((x, y))
        else:
            # Try to reshape if possible
            if landmarks.size % 2 == 0:  # Even number of elements
                reshaped = landmarks.reshape(-1, 2)
                for landmark in reshaped:
                    x = int(landmark[0] * width)
                    y = int(landmark[1] * height)
                    
                    # Clamp to image boundaries
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    
                    points.append((x, y))
            elif landmarks.size % 3 == 0:  # Multiple of 3
                reshaped = landmarks.reshape(-1, 3)
                for landmark in reshaped:
                    x = int(landmark[0] * width)
                    y = int(landmark[1] * height)
                    
                    # Clamp to image boundaries
                    x = max(0, min(x, width - 1))
                    y = max(0, min(y, height - 1))
                    
                    points.append((x, y))
        
        print(f"Extracted {len(points)} landmark points")
        if len(points) > 0:
            print(f"Sample points: {points[:5]}")
        return points


def draw_landmarks(image: np.ndarray, landmarks: List[Tuple[int, int]], 
                  dot_color: Tuple[int, int, int] = (0, 255, 0),
                  dot_size: int = 2) -> np.ndarray:
    """
    Draw landmark dots on the image.
    
    Args:
        image: Input image
        landmarks: List of (x, y) coordinates
        dot_color: Color of the dots in BGR format
        dot_size: Size of the dots
        
    Returns:
        Image with landmarks drawn
    """
    result_image = image.copy()
    
    for i, (x, y) in enumerate(landmarks):
        # Draw filled circle for each landmark
        cv2.circle(result_image, (x, y), dot_size, dot_color, -1)
        
        # Optionally draw landmark number
        if dot_size > 3:
            cv2.putText(result_image, str(i), (x + 3, y - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, dot_color, 1)
    
    return result_image


def detect_face_region(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """
    Detect face region using OpenCV's face detection as fallback.
    
    Args:
        image: Input image
        
    Returns:
        Face bounding box as (x, y, w, h) or None
    """
    try:
        # Load OpenCV's face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        if len(faces) > 0:
            # Return the first detected face
            return tuple(faces[0])
        
        return None
        
    except Exception as e:
        print(f"Error in face detection: {e}")
        return None


def main():
    """Main function to run face landmark detection."""
    parser = argparse.ArgumentParser(description='Face Landmark Detection with ONNX Model')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument('--model', '-m', required=True, help='Path to ONNX model')
    parser.add_argument('--output', '-o', help='Path to output image (optional)')
    parser.add_argument('--dot-size', type=int, default=2, help='Size of landmark dots')
    parser.add_argument('--dot-color', nargs=3, type=int, default=[0, 255, 0], 
                       help='Color of dots in BGR format')
    parser.add_argument('--show-numbers', action='store_true', 
                       help='Show landmark numbers')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        sys.exit(1)
    
    # Load image
    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image '{args.image}'")
        sys.exit(1)
    
    print(f"Image shape: {image.shape}")
    
    # Initialize detector
    print(f"Loading model: {args.model}")
    detector = FaceLandmarkDetector(args.model)
    
    # Detect landmarks
    print("Detecting landmarks...")
    landmarks = detector.detect_landmarks(image)
    
    if landmarks is not None:
        print(f"Landmarks detected! Shape: {landmarks.shape}")
        
        # Convert to pixel coordinates
        landmark_points = detector.postprocess_landmarks(landmarks, image.shape)
        print(f"Number of landmark points: {len(landmark_points)}")
        
        # Draw landmarks
        dot_color = tuple(args.dot_color)  # Convert to tuple
        result_image = draw_landmarks(image, landmark_points, dot_color, args.dot_size)
        
        # Add face bounding box if available
        face_box = detect_face_region(image)
        if face_box:
            x, y, w, h = face_box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            print(f"Face detected at: ({x}, {y}, {w}, {h})")
        
        # Save or display result
        if args.output:
            cv2.imwrite(args.output, result_image)
            print(f"Result saved to: {args.output}")
        else:
            # Generate output filename
            base_name = os.path.splitext(args.image)[0]
            output_path = f"{base_name}_landmarks.jpg"
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to: {output_path}")
        
        # Display result
        print("Displaying result... Press any key to close.")
        cv2.imshow('Face Landmarks', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("No landmarks detected in the image.")
        
        # Try to detect face region as fallback
        face_box = detect_face_region(image)
        if face_box:
            x, y, w, h = face_box
            result_image = image.copy()
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(result_image, "Face detected but no landmarks", 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            
            output_path = f"{os.path.splitext(args.image)[0]}_face_only.jpg"
            cv2.imwrite(output_path, result_image)
            print(f"Face detection result saved to: {output_path}")
            
            cv2.imshow('Face Detection', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No face detected in the image.")


if __name__ == "__main__":
    main() 