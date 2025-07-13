#!/usr/bin/env python3
"""
Face Landmark Detection with Mesh Connections
============================================

This script performs face landmark detection using an ONNX model and draws
both landmark dots and face mesh connections on the input image.

Requirements:
- onnxruntime
- opencv-python
- numpy
- Pillow (PIL)

Usage:
    python face_landmark_detector_with_mesh.py --image A1.jpg --model models/mediapipe_face.onnx
"""

import argparse
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
import sys
from typing import List, Tuple, Optional

# MediaPipe face landmark connections
FACE_LANDMARK_CONNECTIONS = [
    # Lips
    (61, 146), (146, 91), (91, 181), (181, 84), (84, 17), (17, 314), (314, 405), (405, 321),
    (321, 375), (375, 291), (61, 185), (185, 40), (40, 39), (39, 37), (37, 0), (0, 267),
    (267, 269), (269, 270), (270, 409), (409, 291), (78, 95), (95, 88), (88, 178), (178, 87),
    (87, 14), (14, 317), (317, 402), (402, 318), (318, 324), (324, 308), (78, 191), (191, 80),
    (80, 81), (81, 82), (82, 13), (13, 312), (312, 311), (311, 310), (310, 415), (415, 308),
    # Left eye
    (263, 249), (249, 390), (390, 373), (373, 374), (374, 380), (380, 381), (381, 382), (382, 362),
    (263, 466), (466, 388), (388, 387), (387, 386), (386, 385), (385, 384), (384, 398), (398, 362),
    # Left eyebrow
    (276, 283), (283, 282), (282, 295), (295, 285), (300, 293), (293, 334), (334, 296), (296, 336),
    # Right eye
    (33, 7), (7, 163), (163, 144), (144, 145), (145, 153), (153, 154), (154, 155), (155, 133),
    (33, 246), (246, 161), (161, 160), (160, 159), (159, 158), (158, 157), (157, 173), (173, 133),
    # Right eyebrow
    (46, 53), (53, 52), (52, 65), (65, 55), (70, 63), (63, 105), (105, 66), (66, 107),
    # Face oval
    (10, 338), (338, 297), (297, 332), (332, 284), (284, 251), (251, 389), (389, 356), (356, 454),
    (454, 323), (323, 361), (361, 288), (288, 397), (397, 365), (365, 379), (379, 378), (378, 400),
    (400, 377), (377, 152), (152, 148), (148, 176), (176, 149), (149, 150), (150, 136), (136, 172),
    (172, 58), (58, 132), (132, 93), (93, 234), (234, 127), (127, 162), (162, 21), (21, 54),
    (54, 103), (103, 67), (67, 109), (109, 10),
]


class FaceLandmarkDetector:
    """Face landmark detector using ONNX model."""
    
    def __init__(self, model_path: str):
        """Initialize the face landmark detector.
        
        Args:
            model_path: Path to the ONNX model file
        """
        self.model_path = model_path
        self.session = None
        self.input_name = None
        self.output_names = None
        self._load_model()
    
    def _load_model(self):
        """Load the ONNX model."""
        try:
            # Create inference session
            self.session = ort.InferenceSession(self.model_path)
            
            # Get input and output information
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"Model loaded successfully!")
            print(f"Input name: {self.input_name}")
            print(f"Input shape: {self.session.get_inputs()[0].shape}")
            print(f"Output names: {self.output_names}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for model inference.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Preprocessed image ready for inference
        """
        # Get input shape from model
        input_shape = self.session.get_inputs()[0].shape
        batch_size, channels, height, width = input_shape
        
        # Resize image to model input size
        resized = cv2.resize(image, (width, height))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect face landmarks in the image.
        
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
            
            # Process outputs - landmarks are in the second output (index 1)
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
        """Convert normalized landmarks to pixel coordinates.
        
        Args:
            landmarks: Normalized landmarks from model
            original_shape: Original image shape (height, width)
            
        Returns:
            List of (x, y) pixel coordinates
        """
        height, width = original_shape[:2]
        points = []
        
        # MediaPipe face landmarks are typically in format (N, 3) where N is number of landmarks
        # and each landmark has (x, y, z) coordinates normalized to [0, 1]
        if landmarks.ndim == 2 and landmarks.shape[1] == 3:  # Shape: (N, 3)
            for landmark in landmarks:
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                points.append((x, y))
        elif landmarks.ndim == 2 and landmarks.shape[1] == 2:  # Shape: (N, 2)
            for landmark in landmarks:
                x = int(landmark[0] * width)
                y = int(landmark[1] * height)
                points.append((x, y))
        elif landmarks.ndim == 1:  # Flattened array
            # Assume alternating x, y coordinates
            for i in range(0, len(landmarks), 2):
                if i + 1 < len(landmarks):
                    x = int(landmarks[i] * width)
                    y = int(landmarks[i + 1] * height)
                    points.append((x, y))
        else:
            # Try to reshape if possible
            if landmarks.size % 2 == 0:  # Even number of elements
                reshaped = landmarks.reshape(-1, 2)
                for landmark in reshaped:
                    x = int(landmark[0] * width)
                    y = int(landmark[1] * height)
                    points.append((x, y))
            elif landmarks.size % 3 == 0:  # Multiple of 3
                reshaped = landmarks.reshape(-1, 3)
                for landmark in reshaped:
                    x = int(landmark[0] * width)
                    y = int(landmark[1] * height)
                    points.append((x, y))
        
        return points


def draw_landmarks_and_connections(image: np.ndarray, 
                                 landmarks: List[Tuple[int, int]], 
                                 dot_color: Tuple[int, int, int] = (0, 255, 0),
                                 line_color: Tuple[int, int, int] = (255, 0, 0),
                                 dot_size: int = 2,
                                 line_thickness: int = 1) -> np.ndarray:
    """Draw landmarks and face mesh connections on the image.
    
    Args:
        image: Input image
        landmarks: List of (x, y) landmark coordinates
        dot_color: Color for landmark dots in BGR format
        line_color: Color for connection lines in BGR format
        dot_size: Size of landmark dots
        line_thickness: Thickness of connection lines
        
    Returns:
        Image with landmarks and connections drawn
    """
    result_image = image.copy()
    
    # Draw connections first (so they appear behind the dots)
    for connection in FACE_LANDMARK_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(landmarks) and end_idx < len(landmarks):
            start_point = landmarks[start_idx]
            end_point = landmarks[end_idx]
            cv2.line(result_image, start_point, end_point, line_color, line_thickness)
    
    # Draw landmark dots
    for i, (x, y) in enumerate(landmarks):
        # Draw filled circle for each landmark
        cv2.circle(result_image, (x, y), dot_size, dot_color, -1)
    
    return result_image


def main():
    """Main function to run face landmark detection with mesh."""
    parser = argparse.ArgumentParser(description='Face Landmark Detection with Mesh Connections')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument('--model', '-m', required=True, help='Path to ONNX model')
    parser.add_argument('--output', '-o', help='Path to output image (optional)')
    parser.add_argument('--dot-size', type=int, default=2, help='Size of landmark dots')
    parser.add_argument('--line-thickness', type=int, default=1, help='Thickness of connection lines')
    parser.add_argument('--dot-color', nargs=3, type=int, default=[0, 255, 0], 
                       help='Color of dots in BGR format')
    parser.add_argument('--line-color', nargs=3, type=int, default=[255, 0, 0], 
                       help='Color of lines in BGR format')
    
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
        
        # Draw landmarks and connections
        dot_color = tuple(args.dot_color)
        line_color = tuple(args.line_color)
        result_image = draw_landmarks_and_connections(
            image, landmark_points, dot_color, line_color, 
            args.dot_size, args.line_thickness
        )
        
        # Save result
        if args.output:
            cv2.imwrite(args.output, result_image)
            print(f"Result saved to: {args.output}")
        else:
            # Generate output filename
            base_name = os.path.splitext(args.image)[0]
            output_path = f"{base_name}_mesh.jpg"
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to: {output_path}")
        
        # Display result
        print("Displaying result... Press any key to close.")
        cv2.imshow('Face Landmarks with Mesh', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("No landmarks detected in the image.")


if __name__ == "__main__":
    main() 