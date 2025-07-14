#!/usr/bin/env python3
"""
MediaPipe Face Landmark Detection - Final Version
===============================================

This script performs face landmark detection using a MediaPipe ONNX model and adds
landmark dots to the input image. This is the optimized final version.

Requirements:
- onnxruntime
- opencv-python
- numpy
- Pillow (PIL)

Usage:
    python face_landmark_detector_final.py --image A1.jpg
"""

import argparse
import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
import sys
from typing import List, Tuple, Optional

# MediaPipe face landmark connections (from official MediaPipe)
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


class MediaPipeFaceLandmarkDetector:
    """MediaPipe Face landmark detector using ONNX model."""
    
    def __init__(self, model_path: str, verbose: bool = False):
        """Initialize the face landmark detector.
        
        Args:
            model_path: Path to the ONNX model file
            verbose: Whether to print verbose information
        """
        self.model_path = model_path
        self.verbose = verbose
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
            
            if self.verbose:
                print(f"Model loaded successfully!")
                print(f"Input name: {self.input_name}")
                print(f"Input shape: {self.session.get_inputs()[0].shape}")
                print(f"Output names: {self.output_names}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")
    
    def detect_face_region(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect face region using OpenCV's face detector.
        
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
                # Return the largest detected face
                largest_face = max(faces, key=lambda f: f[2] * f[3])
                return tuple(largest_face)
            
            return None
            
        except Exception as e:
            if self.verbose:
                print(f"Error in face detection: {e}")
            return None

    def preprocess_image(self, image: np.ndarray, face_box: Optional[Tuple[int, int, int, int]] = None) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Preprocess image for model inference.
        
        Args:
            image: Input image as numpy array (BGR format)
            face_box: Optional face bounding box (x, y, w, h)
            
        Returns:
            Tuple of (preprocessed image, face_box_used)
        """
        # Get input shape from model (should be [1, 3, 192, 192])
        input_shape = self.session.get_inputs()[0].shape
        batch_size, channels, height, width = input_shape
        
        # If no face box provided, detect it
        if face_box is None:
            face_box = self.detect_face_region(image)
        
        # If still no face detected, use the whole image
        if face_box is None:
            if self.verbose:
                print("No face detected, using full image")
            face_box = (0, 0, image.shape[1], image.shape[0])
        
        # Extract face region with some padding
        x, y, w, h = face_box
        padding = int(max(w, h) * 0.3)  # 30% padding
        
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(image.shape[1], x + w + padding)
        y2 = min(image.shape[0], y + h + padding)
        
        # Crop face region
        face_crop = image[y1:y2, x1:x2]
        
        # Store the actual crop coordinates for later coordinate transformation
        actual_face_box = (x1, y1, x2 - x1, y2 - y1)
        
        # Resize cropped face to model input size
        resized = cv2.resize(face_crop, (width, height))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1] and convert to float32
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Transpose to CHW format and add batch dimension
        transposed = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(transposed, axis=0)
        
        return batched, actual_face_box
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """Detect face landmarks in the image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (landmarks, face_box) or None if no face detected
        """
        try:
            # Preprocess image
            processed_image, face_box = self.preprocess_image(image)
            
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: processed_image})
            
            # Process outputs - landmarks are in the second output (index 1)
            # Output format: ['scores', 'landmarks']
            landmarks = outputs[1] if len(outputs) > 1 else outputs[0]
            
            # Handle batch dimension
            if landmarks.ndim == 3:
                landmarks = landmarks[0]
            
            return landmarks, face_box
            
        except Exception as e:
            if self.verbose:
                print(f"Error during inference: {e}")
            return None
    
    def postprocess_landmarks(self, landmarks: np.ndarray, 
                            face_box: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
        """Convert normalized landmarks to pixel coordinates in original image space.
        
        Args:
            landmarks: Normalized landmarks from model
            face_box: Face bounding box (x, y, w, h) in original image coordinates
            
        Returns:
            List of (x, y) pixel coordinates in original image space
        """
        face_x, face_y, face_w, face_h = face_box
        points = []
        
        # MediaPipe face landmarks are in format (N, 3) where N=468 landmarks
        # The coordinates are normalized to [0,1] relative to the cropped face region
        if landmarks.ndim == 2 and landmarks.shape[1] == 3:  # Shape: (468, 3)
            for landmark in landmarks:
                # Scale normalized coordinates to face crop size
                x_in_crop = landmark[0] * face_w
                y_in_crop = landmark[1] * face_h
                
                # Transform to original image coordinates
                x = int(face_x + x_in_crop)
                y = int(face_y + y_in_crop)
                
                # Clamp to image boundaries (assuming we know the original image size)
                # We'll handle this in the process_image method
                points.append((x, y))
        
        return points
    
    def process_image(self, image: np.ndarray) -> Optional[List[Tuple[int, int]]]:
        """Complete pipeline to process image and get landmark points.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of (x, y) landmark coordinates or None if no face detected
        """
        # Detect landmarks
        result = self.detect_landmarks(image)
        
        if result is not None:
            landmarks, face_box = result
            
            # Convert to pixel coordinates
            landmark_points = self.postprocess_landmarks(landmarks, face_box)
            
            # Clamp points to image boundaries
            height, width = image.shape[:2]
            clamped_points = []
            for x, y in landmark_points:
                x = max(0, min(x, width - 1))
                y = max(0, min(y, height - 1))
                clamped_points.append((x, y))
            
            if self.verbose:
                print(f"Detected {len(clamped_points)} landmark points")
                print(f"Face box: {face_box}")
            
            return clamped_points
        
        return None


def draw_landmarks(image: np.ndarray, 
                  landmarks: List[Tuple[int, int]], 
                  dot_color: Tuple[int, int, int] = (0, 255, 0),
                  dot_size: int = 2) -> np.ndarray:
    """Draw landmark dots on the image.
    
    Args:
        image: Input image
        landmarks: List of (x, y) landmark coordinates
        dot_color: Color for landmark dots in BGR format
        dot_size: Size of landmark dots
        
    Returns:
        Image with landmarks drawn
    """
    result_image = image.copy()
    
    for x, y in landmarks:
        cv2.circle(result_image, (x, y), dot_size, dot_color, -1)
    
    return result_image


def draw_face_mesh(image: np.ndarray, 
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
        Image with face mesh drawn
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
    for x, y in landmarks:
        cv2.circle(result_image, (x, y), dot_size, dot_color, -1)
    
    return result_image


def detect_face_landmarks(image_path: str, 
                         model_path: str = 'models/mediapipe_face.onnx',
                         output_path: Optional[str] = None,
                         draw_mesh: bool = False,
                         dot_size: int = 2,
                         dot_color: Tuple[int, int, int] = (0, 255, 0),
                         line_color: Tuple[int, int, int] = (255, 0, 0),
                         line_thickness: int = 1,
                         verbose: bool = False) -> Optional[str]:
    """
    Detect face landmarks and save the result image.
    
    Args:
        image_path: Path to input image
        model_path: Path to ONNX model (default: 'models/mediapipe_face.onnx')
        output_path: Path to output image (optional, auto-generated if None)
        draw_mesh: Whether to draw face mesh connections
        dot_size: Size of landmark dots
        dot_color: Color of dots in BGR format
        line_color: Color of mesh lines in BGR format
        line_thickness: Thickness of mesh lines
        verbose: Whether to print verbose output
        
    Returns:
        Path to the output image if successful, None otherwise
    """
    # Check if files exist
    if not os.path.exists(image_path):
        if verbose:
            print(f"Error: Image file '{image_path}' not found!")
        return None
    
    if not os.path.exists(model_path):
        if verbose:
            print(f"Error: Model file '{model_path}' not found!")
        return None
    
    # Load image
    if verbose:
        print(f"Loading image: {image_path}")
    
    image = cv2.imread(image_path)
    if image is None:
        if verbose:
            print(f"Error: Could not load image '{image_path}'")
        return None
    
    # Initialize detector
    if verbose:
        print(f"Loading model: {model_path}")
    
    detector = MediaPipeFaceLandmarkDetector(model_path, verbose=verbose)
    
    # Process image
    if verbose:
        print("Detecting landmarks...")
    
    landmark_points = detector.process_image(image)
    
    if landmark_points is not None:
        if verbose:
            print(f"Successfully detected {len(landmark_points)} landmarks")
        
        # Draw landmarks
        if draw_mesh:
            result_image = draw_face_mesh(
                image, landmark_points, dot_color, line_color, 
                dot_size, line_thickness
            )
        else:
            result_image = draw_landmarks(image, landmark_points, dot_color, dot_size)
        
        # Generate output path if not provided
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            suffix = "_mesh" if draw_mesh else "_landmarks"
            output_path = f"{base_name}{suffix}.jpg"
        
        # Save result
        cv2.imwrite(output_path, result_image)
        if verbose:
            print(f"Result saved to: {output_path}")
        
        return output_path
    
    else:
        if verbose:
            print("No landmarks detected in the image.")
        return None


def main():
    """Main function to run face landmark detection."""
    parser = argparse.ArgumentParser(description='MediaPipe Face Landmark Detection')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument('--model', '-m', default='models/mediapipe_face.onnx', help='Path to ONNX model (default: models/mediapipe_face.onnx)')
    parser.add_argument('--output', '-o', help='Path to output image (optional)')
    parser.add_argument('--dot-size', type=int, default=2, help='Size of landmark dots')
    parser.add_argument('--dot-color', nargs=3, type=int, default=[0, 255, 0], 
                       help='Color of dots in BGR format')
    parser.add_argument('--mesh', action='store_true', help='Draw face mesh connections')
    parser.add_argument('--line-color', nargs=3, type=int, default=[255, 0, 0], 
                       help='Color of mesh lines in BGR format')
    parser.add_argument('--line-thickness', type=int, default=1, help='Thickness of mesh lines')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-display', action='store_true', help='Don\'t display result window')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        print(f"Please ensure the MediaPipe face model is at: {args.model}")
        sys.exit(1)
    
    # Load image
    if args.verbose:
        print(f"Loading image: {args.image}")
    
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image '{args.image}'")
        sys.exit(1)
    
    if args.verbose:
        print(f"Image shape: {image.shape}")
    
    # Initialize detector
    if args.verbose:
        print(f"Loading model: {args.model}")
    
    detector = MediaPipeFaceLandmarkDetector(args.model, verbose=args.verbose)
    
    # Process image
    if args.verbose:
        print("Detecting landmarks...")
    
    landmark_points = detector.process_image(image)
    
    if landmark_points is not None:
        if args.verbose:
            print(f"Successfully detected {len(landmark_points)} landmarks")
        
        # Draw landmarks
        dot_color = tuple(args.dot_color)
        
        if args.mesh:
            line_color = tuple(args.line_color)
            result_image = draw_face_mesh(
                image, landmark_points, dot_color, line_color, 
                args.dot_size, args.line_thickness
            )
        else:
            result_image = draw_landmarks(image, landmark_points, dot_color, args.dot_size)
        
        # Save result
        if args.output:
            output_path = args.output
        else:
            # Generate output filename
            base_name = os.path.splitext(args.image)[0]
            suffix = "_mesh" if args.mesh else "_landmarks"
            output_path = f"{base_name}{suffix}.jpg"
        
        cv2.imwrite(output_path, result_image)
        print(f"Result saved to: {output_path}")
        
        # Display result
        if not args.no_display:
            window_name = 'Face Mesh' if args.mesh else 'Face Landmarks'
            cv2.imshow(window_name, result_image)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
    else:
        print("No landmarks detected in the image.")
        sys.exit(1)


if __name__ == "__main__":
    main() 