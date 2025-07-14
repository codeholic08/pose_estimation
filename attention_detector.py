#!/usr/bin/env python3
"""
Attention Detection System
=========================

This script analyzes face landmarks to detect attention using multiple factors:
1. Nose-to-eye distance symmetry (head turning)
2. Eye aspect ratio (eye openness)
3. Gaze direction analysis
4. Facial symmetry

Usage:
    python attention_detector.py --image A1.jpg
"""

import argparse
import cv2
import numpy as np
import math
import os
import sys
from typing import List, Tuple, Optional, Dict
from face_landmark_detector_final import MediaPipeFaceLandmarkDetector

# MediaPipe face landmark indices
NOSE_TIP = 1          # Nose tip
NOSE_TIP_ACTUAL = 4   # More accurate nose tip
NOSE_BRIDGE = 6       # Nose bridge point

# Eye landmark points for more accurate analysis
LEFT_EYE_POINTS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_POINTS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]

# Eye aspect ratio points (for eye openness detection)
LEFT_EYE_VERTICAL = [159, 145]  # Top and bottom of left eye
LEFT_EYE_HORIZONTAL = [33, 133]  # Left and right corners of left eye
RIGHT_EYE_VERTICAL = [386, 374]  # Top and bottom of right eye  
RIGHT_EYE_HORIZONTAL = [362, 263]  # Left and right corners of right eye

# Face outline points for symmetry analysis
FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

class AttentionDetector:
    """Advanced attention detector using multiple facial analysis factors."""
    
    def __init__(self, model_path: str = 'models/mediapipe_face.onnx', verbose: bool = False):
        """Initialize the attention detector.
        
        Args:
            model_path: Path to the MediaPipe ONNX model
            verbose: Whether to print verbose output
        """
        self.verbose = verbose
        self.face_detector = MediaPipeFaceLandmarkDetector(model_path, verbose=verbose)
        
    def calculate_eye_center(self, landmarks: List[Tuple[int, int]], eye_points: List[int]) -> Tuple[float, float]:
        """Calculate the center of an eye using multiple landmark points.
        
        Args:
            landmarks: List of all face landmarks
            eye_points: List of landmark indices for the eye
            
        Returns:
            (x, y) coordinates of the eye center
        """
        if len(landmarks) <= max(eye_points):
            return (0.0, 0.0)
        
        x_coords = [landmarks[i][0] for i in eye_points if i < len(landmarks)]
        y_coords = [landmarks[i][1] for i in eye_points if i < len(landmarks)]
        
        if not x_coords or not y_coords:
            return (0.0, 0.0)
        
        center_x = sum(x_coords) / len(x_coords)
        center_y = sum(y_coords) / len(y_coords)
        
        return (center_x, center_y)
    
    def calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def calculate_eye_aspect_ratio(self, landmarks: List[Tuple[int, int]], 
                                 vertical_points: List[int], horizontal_points: List[int]) -> float:
        """Calculate eye aspect ratio to determine eye openness.
        
        Args:
            landmarks: List of all face landmarks
            vertical_points: Indices for vertical eye measurements
            horizontal_points: Indices for horizontal eye measurements
            
        Returns:
            Eye aspect ratio (higher = more open)
        """
        if len(landmarks) <= max(vertical_points + horizontal_points):
            return 0.0
        
        # Calculate vertical distance (eye height)
        vertical_dist = self.calculate_distance(
            landmarks[vertical_points[0]], landmarks[vertical_points[1]]
        )
        
        # Calculate horizontal distance (eye width)
        horizontal_dist = self.calculate_distance(
            landmarks[horizontal_points[0]], landmarks[horizontal_points[1]]
        )
        
        # Aspect ratio = height / width
        if horizontal_dist > 0:
            return vertical_dist / horizontal_dist
        return 0.0
    
    def analyze_facial_symmetry(self, landmarks: List[Tuple[int, int]]) -> float:
        """Analyze facial symmetry to detect head pose.
        
        Args:
            landmarks: List of all face landmarks
            
        Returns:
            Symmetry score (0.0 to 1.0, higher = more symmetric)
        """
        if len(landmarks) < 50:
            return 0.5
        
        # Get nose bridge as reference point
        nose_bridge = landmarks[NOSE_BRIDGE] if NOSE_BRIDGE < len(landmarks) else landmarks[0]
        
        # Calculate distances from nose to key facial points
        left_distances = []
        right_distances = []
        
        # Sample some key points for symmetry analysis
        key_points = [33, 133, 362, 263, 61, 291, 39, 269, 21, 251]  # Various face points
        
        for i in range(0, len(key_points), 2):
            if i + 1 < len(key_points):
                left_idx = key_points[i]
                right_idx = key_points[i + 1]
                
                if left_idx < len(landmarks) and right_idx < len(landmarks):
                    left_dist = self.calculate_distance(nose_bridge, landmarks[left_idx])
                    right_dist = self.calculate_distance(nose_bridge, landmarks[right_idx])
                    
                    left_distances.append(left_dist)
                    right_distances.append(right_dist)
        
        if not left_distances or not right_distances:
            return 0.5
        
        # Calculate symmetry score
        symmetry_scores = []
        for left_dist, right_dist in zip(left_distances, right_distances):
            if max(left_dist, right_dist) > 0:
                symmetry = min(left_dist, right_dist) / max(left_dist, right_dist)
                symmetry_scores.append(symmetry)
        
        return sum(symmetry_scores) / len(symmetry_scores) if symmetry_scores else 0.5
    
    def analyze_attention(self, landmarks: List[Tuple[int, int]]) -> Dict:
        """Analyze attention using multiple factors.
        
        Args:
            landmarks: List of face landmarks
            
        Returns:
            Dictionary containing comprehensive attention analysis
        """
        if len(landmarks) < 10:
            return {
                'attention_score': 0.0,
                'is_attentive': False,
                'head_direction': 'unknown',
                'error': 'Insufficient landmarks'
            }
        
        # Factor 1: Nose-to-eye distance analysis (head turning)
        nose_point = landmarks[NOSE_TIP_ACTUAL] if NOSE_TIP_ACTUAL < len(landmarks) else landmarks[0]
        left_eye_center = self.calculate_eye_center(landmarks, LEFT_EYE_POINTS)
        right_eye_center = self.calculate_eye_center(landmarks, RIGHT_EYE_POINTS)
        
        if left_eye_center == (0.0, 0.0) or right_eye_center == (0.0, 0.0):
            return {
                'attention_score': 0.0,
                'is_attentive': False,
                'head_direction': 'unknown',
                'error': 'Could not calculate eye centers'
            }
        
        nose_to_left_eye = self.calculate_distance(nose_point, left_eye_center)
        nose_to_right_eye = self.calculate_distance(nose_point, right_eye_center)
        
        # Calculate head turning score
        distance_difference = abs(nose_to_left_eye - nose_to_right_eye)
        average_distance = (nose_to_left_eye + nose_to_right_eye) / 2
        normalized_difference = distance_difference / average_distance if average_distance > 0 else 1.0
        head_turning_score = max(0.0, 1.0 - normalized_difference)
        
        # Factor 2: Eye aspect ratio (eye openness)
        left_ear = self.calculate_eye_aspect_ratio(landmarks, LEFT_EYE_VERTICAL, LEFT_EYE_HORIZONTAL)
        right_ear = self.calculate_eye_aspect_ratio(landmarks, RIGHT_EYE_VERTICAL, RIGHT_EYE_HORIZONTAL)
        
        # Average eye aspect ratio
        avg_ear = (left_ear + right_ear) / 2
        
        # Eye openness score (normalized)
        # Typical EAR for open eyes is around 0.2-0.3, closed eyes ~0.1
        # Penalize extremely high EAR values as they might indicate unnatural state
        base_openness = min(1.0, max(0.0, (avg_ear - 0.1) / 0.2))
        
        # Penalize very high eye openness (> 0.35 EAR) as potentially unnatural
        if avg_ear > 0.35:
            penalty = (avg_ear - 0.35) * 2.0  # Penalty increases with extreme openness
            eye_openness_score = max(0.0, base_openness - penalty)
        else:
            eye_openness_score = base_openness
        
        # Factor 3: Eye symmetry (one eye more closed than other)
        ear_difference = abs(left_ear - right_ear)
        eye_symmetry_score = max(0.0, 1.0 - (ear_difference / 0.1))  # Normalize by expected variance
        
        # Factor 4: Facial symmetry (head pose)
        facial_symmetry_score = self.analyze_facial_symmetry(landmarks)
        
        # Combined attention score with weights
        attention_score = (
            head_turning_score * 0.25 +     # 25% weight for head turning
            eye_openness_score * 0.35 +     # 35% weight for eye openness
            eye_symmetry_score * 0.35 +     # 35% weight for eye symmetry (very important)
            facial_symmetry_score * 0.05    # 5% weight for facial symmetry
        )
        
        # Determine head direction
        head_direction = 'straight'
        if nose_to_left_eye > nose_to_right_eye * 1.15:
            head_direction = 'turned_right'
        elif nose_to_right_eye > nose_to_left_eye * 1.15:
            head_direction = 'turned_left'
        
        # More strict threshold for attention - adjusted based on data analysis
        is_attentive = attention_score > 0.85  # 85% threshold
        
        return {
            'attention_score': attention_score,
            'is_attentive': is_attentive,
            'head_direction': head_direction,
            'nose_to_left_eye': nose_to_left_eye,
            'nose_to_right_eye': nose_to_right_eye,
            'distance_difference': distance_difference,
            'normalized_difference': normalized_difference,
            'head_turning_score': head_turning_score,
            'left_ear': left_ear,
            'right_ear': right_ear,
            'avg_ear': avg_ear,
            'eye_openness_score': eye_openness_score,
            'eye_symmetry_score': eye_symmetry_score,
            'facial_symmetry_score': facial_symmetry_score,
            'nose_point': nose_point,
            'left_eye_center': left_eye_center,
            'right_eye_center': right_eye_center
        }
    
    def draw_attention_visualization(self, image: np.ndarray, 
                                   landmarks: List[Tuple[int, int]], 
                                   attention_data: Dict) -> np.ndarray:
        """Draw attention analysis visualization on the image.
        
        Args:
            image: Input image
            landmarks: Face landmarks
            attention_data: Attention analysis results
            
        Returns:
            Image with attention visualization
        """
        result_image = image.copy()
        
        if 'error' in attention_data:
            # Draw error message
            cv2.putText(result_image, f"Error: {attention_data['error']}", 
                       (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return result_image
        
        # Extract data
        nose_point = attention_data['nose_point']
        left_eye_center = attention_data['left_eye_center']
        right_eye_center = attention_data['right_eye_center']
        attention_score = attention_data['attention_score']
        is_attentive = attention_data['is_attentive']
        head_direction = attention_data['head_direction']
        
        # Convert float coordinates to int
        nose_point = (int(nose_point[0]), int(nose_point[1]))
        left_eye_center = (int(left_eye_center[0]), int(left_eye_center[1]))
        right_eye_center = (int(right_eye_center[0]), int(right_eye_center[1]))
        
        # Draw nose point
        cv2.circle(result_image, nose_point, 5, (0, 255, 255), -1)  # Yellow
        
        # Draw eye centers
        cv2.circle(result_image, left_eye_center, 5, (255, 0, 0), -1)  # Blue
        cv2.circle(result_image, right_eye_center, 5, (255, 0, 0), -1)  # Blue
        
        # Draw distance lines
        line_color = (0, 255, 0) if is_attentive else (0, 0, 255)  # Green if attentive, red if not
        cv2.line(result_image, nose_point, left_eye_center, line_color, 2)
        cv2.line(result_image, nose_point, right_eye_center, line_color, 2)
        
        # Draw distance values
        left_distance = attention_data['nose_to_left_eye']
        right_distance = attention_data['nose_to_right_eye']
        
        # Position text near the lines
        left_mid = ((nose_point[0] + left_eye_center[0]) // 2, (nose_point[1] + left_eye_center[1]) // 2)
        right_mid = ((nose_point[0] + right_eye_center[0]) // 2, (nose_point[1] + right_eye_center[1]) // 2)
        
        cv2.putText(result_image, f"{left_distance:.1f}", left_mid, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result_image, f"{right_distance:.1f}", right_mid, 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw attention status
        status_color = (0, 255, 0) if is_attentive else (0, 0, 255)
        status_text = "ATTENTIVE" if is_attentive else "NOT ATTENTIVE"
        
        cv2.putText(result_image, status_text, (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Draw attention score
        cv2.putText(result_image, f"Score: {attention_score:.2f}", (50, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw head direction
        direction_text = f"Head: {head_direction.replace('_', ' ').title()}"
        cv2.putText(result_image, direction_text, (50, 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw detailed metrics
        y_offset = 170
        metrics = [
            f"Head Turn: {attention_data['head_turning_score']:.2f}",
            f"Eye Open: {attention_data['eye_openness_score']:.2f}",
            f"Eye Sym: {attention_data['eye_symmetry_score']:.2f}",
            f"Face Sym: {attention_data['facial_symmetry_score']:.2f}",
            f"EAR: {attention_data['avg_ear']:.3f}"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(result_image, metric, (50, y_offset + i * 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return result_image
    
    def process_image(self, image_path: str, output_path: Optional[str] = None) -> Optional[Dict]:
        """Process an image and analyze attention.
        
        Args:
            image_path: Path to input image
            output_path: Path to output image (optional)
            
        Returns:
            Attention analysis results or None if failed
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            if self.verbose:
                print(f"Error: Could not load image '{image_path}'")
            return None
        
        # Detect landmarks
        landmark_points = self.face_detector.process_image(image)
        
        if landmark_points is None:
            if self.verbose:
                print("No face landmarks detected")
            return None
        
        # Analyze attention
        attention_data = self.analyze_attention(landmark_points)
        
        # Create visualization
        result_image = self.draw_attention_visualization(image, landmark_points, attention_data)
        
        # Save result
        if output_path is None:
            base_name = os.path.splitext(image_path)[0]
            output_path = f"{base_name}_attention.jpg"
        
        cv2.imwrite(output_path, result_image)
        
        if self.verbose:
            print(f"Attention analysis saved to: {output_path}")
            if 'error' not in attention_data:
                print(f"Attention Score: {attention_data['attention_score']:.2f}")
                print(f"Is Attentive: {attention_data['is_attentive']}")
                print(f"Head Direction: {attention_data['head_direction']}")
                print(f"Nose to Left Eye: {attention_data['nose_to_left_eye']:.1f}")
                print(f"Nose to Right Eye: {attention_data['nose_to_right_eye']:.1f}")
        
        return attention_data


def main():
    """Main function to run attention detection."""
    parser = argparse.ArgumentParser(description='Attention Detection System')
    parser.add_argument('--image', '-i', required=True, help='Path to input image')
    parser.add_argument('--model', '-m', default='models/mediapipe_face.onnx', 
                       help='Path to ONNX model (default: models/mediapipe_face.onnx)')
    parser.add_argument('--output', '-o', help='Path to output image (optional)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-display', action='store_true', help='Don\'t display result window')
    
    args = parser.parse_args()
    
    # Check if files exist
    if not os.path.exists(args.image):
        print(f"Error: Image file '{args.image}' not found!")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        sys.exit(1)
    
    # Initialize detector
    detector = AttentionDetector(args.model, verbose=args.verbose)
    
    # Process image
    attention_data = detector.process_image(args.image, args.output)
    
    if attention_data is None:
        print("Failed to analyze attention")
        sys.exit(1)
    
    # Display result
    if not args.no_display:
        output_path = args.output if args.output else f"{os.path.splitext(args.image)[0]}_attention.jpg"
        if os.path.exists(output_path):
            result_image = cv2.imread(output_path)
            cv2.imshow('Attention Detection', result_image)
            print("Press any key to close the window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main() 