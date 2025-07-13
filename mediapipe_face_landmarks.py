#!/usr/bin/env python3
"""
MediaPipe Face Landmark Detection
=================================

This script uses MediaPipe's Python API to detect face landmarks and add
landmark dots to the input image. This is an alternative to the ONNX approach
and may be more compatible with MediaPipe models.

Requirements:
- mediapipe
- opencv-python
- numpy

Usage:
    python mediapipe_face_landmarks.py --image A1.jpg
"""

import argparse
import cv2
import numpy as np
import mediapipe as mp
import os
import sys
from typing import List, Tuple, Optional


class MediaPipeFaceLandmarks:
    """Face landmark detector using MediaPipe."""
    
    def __init__(self, 
                 static_image_mode: bool = True,
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5):
        """
        Initialize MediaPipe face mesh detector.
        
        Args:
            static_image_mode: Whether to treat input as static images
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks around eyes and lips
            min_detection_confidence: Minimum confidence for face detection
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence
        )
    
    def detect_landmarks(self, image: np.ndarray) -> Optional[List]:
        """
        Detect face landmarks in the image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of face landmarks or None if no face detected
        """
        try:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image
            results = self.face_mesh.process(rgb_image)
            
            return results.multi_face_landmarks
            
        except Exception as e:
            print(f"Error during landmark detection: {e}")
            return None
    
    def extract_landmark_points(self, landmarks, image_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Extract landmark points as pixel coordinates.
        
        Args:
            landmarks: MediaPipe landmarks
            image_shape: Image shape (height, width)
            
        Returns:
            List of (x, y) pixel coordinates
        """
        height, width = image_shape[:2]
        points = []
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))
        
        return points


def draw_landmarks_simple(image: np.ndarray, landmarks: List[Tuple[int, int]], 
                         dot_color: Tuple[int, int, int] = (0, 255, 0),
                         dot_size: int = 1) -> np.ndarray:
    """
    Draw simple landmark dots on the image.
    
    Args:
        image: Input image
        landmarks: List of (x, y) coordinates
        dot_color: Color of the dots in BGR format
        dot_size: Size of the dots
        
    Returns:
        Image with landmarks drawn
    """
    result_image = image.copy()
    
    for x, y in landmarks:
        cv2.circle(result_image, (x, y), dot_size, dot_color, -1)
    
    return result_image


def draw_landmarks_with_connections(image: np.ndarray, landmarks, 
                                  mp_face_mesh, mp_drawing, mp_drawing_styles) -> np.ndarray:
    """
    Draw landmarks with MediaPipe's default connections.
    
    Args:
        image: Input image
        landmarks: MediaPipe landmarks
        mp_face_mesh: MediaPipe face mesh module
        mp_drawing: MediaPipe drawing utilities
        mp_drawing_styles: MediaPipe drawing styles
        
    Returns:
        Image with landmarks and connections drawn
    """
    result_image = image.copy()
    
    # Draw face mesh tesselation
    mp_drawing.draw_landmarks(
        image=result_image,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
    )
    
    # Draw face mesh contours
    mp_drawing.draw_landmarks(
        image=result_image,
        landmark_list=landmarks,
        connections=mp_face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
    )
    
    # Draw irises if refined landmarks are available
    if len(landmarks.landmark) > 468:
        mp_drawing.draw_landmarks(
            image=result_image,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        )
    
    return result_image


def main():
    """Main function to run face landmark detection."""
    parser = argparse.ArgumentParser(description='MediaPipe Face Landmark Detection')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument('--output', '-o', help='Path to output image (optional)')
    parser.add_argument('--dot-size', type=int, default=1, help='Size of landmark dots')
    parser.add_argument('--dot-color', nargs=3, type=int, default=[0, 255, 0], 
                       help='Color of dots in BGR format')
    parser.add_argument('--style', choices=['dots', 'mesh', 'both'], default='dots',
                       help='Drawing style: dots only, mesh only, or both')
    parser.add_argument('--max-faces', type=int, default=1, help='Maximum number of faces to detect')
    parser.add_argument('--confidence', type=float, default=0.5, help='Minimum detection confidence')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        sys.exit(1)
    
    # Load image
    print(f"Loading image: {args.image}")
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Could not load image '{args.image}'")
        sys.exit(1)
    
    print(f"Image shape: {image.shape}")
    
    # Initialize detector
    print("Initializing MediaPipe face mesh detector...")
    detector = MediaPipeFaceLandmarks(
        max_num_faces=args.max_faces,
        min_detection_confidence=args.confidence
    )
    
    # Detect landmarks
    print("Detecting landmarks...")
    face_landmarks = detector.detect_landmarks(image)
    
    if face_landmarks:
        print(f"Detected {len(face_landmarks)} face(s)")
        
        result_image = image.copy()
        
        for face_idx, landmarks in enumerate(face_landmarks):
            print(f"Processing face {face_idx + 1} with {len(landmarks.landmark)} landmarks")
            
            if args.style in ['dots', 'both']:
                # Extract landmark points
                landmark_points = detector.extract_landmark_points(landmarks, image.shape)
                
                # Draw simple dots
                dot_color = tuple(args.dot_color)
                result_image = draw_landmarks_simple(result_image, landmark_points, 
                                                   dot_color, args.dot_size)
            
            if args.style in ['mesh', 'both']:
                # Draw mesh with connections
                result_image = draw_landmarks_with_connections(
                    result_image, landmarks, 
                    detector.mp_face_mesh, detector.mp_drawing, detector.mp_drawing_styles
                )
        
        # Save or display result
        if args.output:
            cv2.imwrite(args.output, result_image)
            print(f"Result saved to: {args.output}")
        else:
            # Generate output filename
            base_name = os.path.splitext(args.image)[0]
            output_path = f"{base_name}_mediapipe_landmarks.jpg"
            cv2.imwrite(output_path, result_image)
            print(f"Result saved to: {output_path}")
        
        # Display result
        print("Displaying result... Press any key to close.")
        cv2.imshow('MediaPipe Face Landmarks', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        print("No face landmarks detected in the image.")
        print("This could be due to:")
        print("- No face visible in the image")
        print("- Face is too small or unclear")
        print("- Detection confidence threshold is too high")
        print("Try adjusting the --confidence parameter (lower values like 0.3)")


if __name__ == "__main__":
    main() 