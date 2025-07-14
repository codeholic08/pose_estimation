#!/usr/bin/env python3
"""
Real-time Webcam Attention Detection System
==========================================

This script captures video from your webcam and analyzes attention in real-time using
the multi-factor attention detection algorithm.

Features:
- Real-time face landmark detection
- Live attention scoring with visual feedback
- Attention statistics tracking
- Keyboard controls for interaction

Controls:
- 'q' or ESC: Quit the application
- 's': Save current frame with attention analysis
- 'r': Reset statistics
- SPACE: Pause/Resume

Usage:
    python webcam_attention_detector.py
"""

import cv2
import numpy as np
import time
import os
from datetime import datetime
from typing import Dict, List, Optional
from attention_detector import AttentionDetector

class WebcamAttentionDetector:
    """Real-time webcam attention detection system."""
    
    def __init__(self, model_path: str = 'models/mediapipe_face.onnx'):
        """Initialize the webcam attention detector.
        
        Args:
            model_path: Path to the MediaPipe ONNX model
        """
        self.attention_detector = AttentionDetector(model_path, verbose=False)
        self.cap = None
        self.is_running = False
        self.is_paused = False
        
        # Statistics tracking
        self.frame_count = 0
        self.attention_scores = []
        self.attentive_frames = 0
        self.start_time = None
        
        # Display settings
        self.window_name = "Real-time Attention Detection"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.thickness = 2
        
        # Colors
        self.green = (0, 255, 0)
        self.red = (0, 0, 255)
        self.blue = (255, 0, 0)
        self.yellow = (0, 255, 255)
        self.white = (255, 255, 255)
        self.black = (0, 0, 0)
        
    def initialize_camera(self, camera_index: int = 0) -> bool:
        """Initialize the camera.
        
        Args:
            camera_index: Camera index (0 for default camera)
            
        Returns:
            True if camera initialized successfully, False otherwise
        """
        # Try different camera backends for better compatibility
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY]  # DirectShow for Windows, then any available
        
        for backend in backends:
            try:
                self.cap = cv2.VideoCapture(camera_index, backend)
                if self.cap.isOpened():
                    break
            except:
                continue
        
        if not self.cap or not self.cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            print("Try checking:")
            print("  - Camera is connected and not used by another application")
            print("  - Camera drivers are installed")
            print("  - Try different camera index (0, 1, 2, etc.)")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Test if we can read a frame
        ret, test_frame = self.cap.read()
        if not ret:
            print("Error: Could not read frame from camera")
            return False
        
        print(f"Camera {camera_index} initialized successfully")
        print(f"Frame size: {test_frame.shape[1]}x{test_frame.shape[0]}")
        return True
    
    def draw_attention_overlay(self, frame: np.ndarray, attention_data: Dict) -> np.ndarray:
        """Draw attention analysis overlay on the frame.
        
        Args:
            frame: Input frame
            attention_data: Attention analysis results
            
        Returns:
            Frame with attention overlay
        """
        overlay = frame.copy()
        
        if 'error' in attention_data:
            # Draw error message
            cv2.putText(overlay, f"Error: {attention_data['error']}", 
                       (20, 30), self.font, self.font_scale, self.red, self.thickness)
            return overlay
        
        # Extract attention data
        attention_score = attention_data['attention_score']
        is_attentive = attention_data['is_attentive']
        head_direction = attention_data['head_direction']
        
        # Draw main attention status
        status_text = "ATTENTIVE" if is_attentive else "NOT ATTENTIVE"
        status_color = self.green if is_attentive else self.red
        
        # Create a semi-transparent background for better text visibility
        overlay_bg = overlay.copy()
        cv2.rectangle(overlay_bg, (10, 10), (300, 200), self.black, -1)
        cv2.addWeighted(overlay_bg, 0.3, overlay, 0.7, 0, overlay)
        
        # Draw attention status
        cv2.putText(overlay, status_text, (20, 40), 
                   self.font, 1.0, status_color, self.thickness)
        
        # Draw attention score
        cv2.putText(overlay, f"Score: {attention_score:.2f}", (20, 70), 
                   self.font, self.font_scale, self.white, self.thickness)
        
        # Draw head direction
        cv2.putText(overlay, f"Head: {head_direction.replace('_', ' ').title()}", (20, 100), 
                   self.font, self.font_scale, self.white, self.thickness)
        
        # Draw detailed metrics
        metrics = [
            f"Head Turn: {attention_data['head_turning_score']:.2f}",
            f"Eye Open: {attention_data['eye_openness_score']:.2f}",
            f"Eye Sym: {attention_data['eye_symmetry_score']:.2f}",
            f"Face Sym: {attention_data['facial_symmetry_score']:.2f}"
        ]
        
        for i, metric in enumerate(metrics):
            cv2.putText(overlay, metric, (20, 130 + i * 20), 
                       self.font, 0.4, self.white, 1)
        
        # Draw landmarks and connections if available
        if 'nose_point' in attention_data:
            nose_point = attention_data['nose_point']
            left_eye_center = attention_data['left_eye_center']
            right_eye_center = attention_data['right_eye_center']
            
            # Convert to int coordinates
            nose_point = (int(nose_point[0]), int(nose_point[1]))
            left_eye_center = (int(left_eye_center[0]), int(left_eye_center[1]))
            right_eye_center = (int(right_eye_center[0]), int(right_eye_center[1]))
            
            # Draw points
            cv2.circle(overlay, nose_point, 3, self.yellow, -1)
            cv2.circle(overlay, left_eye_center, 3, self.blue, -1)
            cv2.circle(overlay, right_eye_center, 3, self.blue, -1)
            
            # Draw lines
            line_color = self.green if is_attentive else self.red
            cv2.line(overlay, nose_point, left_eye_center, line_color, 1)
            cv2.line(overlay, nose_point, right_eye_center, line_color, 1)
        
        return overlay
    
    def draw_statistics(self, frame: np.ndarray) -> np.ndarray:
        """Draw session statistics on the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with statistics overlay
        """
        if self.frame_count == 0:
            return frame
        
        # Calculate statistics
        avg_score = sum(self.attention_scores) / len(self.attention_scores) if self.attention_scores else 0
        attention_percentage = (self.attentive_frames / self.frame_count) * 100
        
        # Calculate session duration
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)
        
        # Draw statistics background
        stats_bg = frame.copy()
        cv2.rectangle(stats_bg, (frame.shape[1] - 250, 10), (frame.shape[1] - 10, 150), self.black, -1)
        cv2.addWeighted(stats_bg, 0.3, frame, 0.7, 0, frame)
        
        # Draw statistics text
        stats_x = frame.shape[1] - 240
        cv2.putText(frame, "SESSION STATS", (stats_x, 30), 
                   self.font, 0.5, self.white, 1)
        cv2.putText(frame, f"Duration: {minutes:02d}:{seconds:02d}", (stats_x, 50), 
                   self.font, 0.4, self.white, 1)
        cv2.putText(frame, f"Frames: {self.frame_count}", (stats_x, 70), 
                   self.font, 0.4, self.white, 1)
        cv2.putText(frame, f"Avg Score: {avg_score:.2f}", (stats_x, 90), 
                   self.font, 0.4, self.white, 1)
        cv2.putText(frame, f"Attentive: {attention_percentage:.1f}%", (stats_x, 110), 
                   self.font, 0.4, self.white, 1)
        
        return frame
    
    def draw_controls(self, frame: np.ndarray) -> np.ndarray:
        """Draw control instructions on the frame.
        
        Args:
            frame: Input frame
            
        Returns:
            Frame with control instructions
        """
        controls = [
            "Controls:",
            "Q/ESC: Quit",
            "S: Save frame",
            "R: Reset stats",
            "SPACE: Pause/Resume"
        ]
        
        # Draw controls background
        controls_bg = frame.copy()
        cv2.rectangle(controls_bg, (10, frame.shape[0] - 120), (200, frame.shape[0] - 10), self.black, -1)
        cv2.addWeighted(controls_bg, 0.3, frame, 0.7, 0, frame)
        
        # Draw controls text
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (20, frame.shape[0] - 100 + i * 15), 
                       self.font, 0.4, self.white, 1)
        
        return frame
    
    def save_frame(self, frame: np.ndarray, attention_data: Dict) -> None:
        """Save the current frame with attention analysis.
        
        Args:
            frame: Current frame
            attention_data: Attention analysis results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"webcam_attention_{timestamp}.jpg"
        
        # Create annotated frame
        annotated_frame = self.draw_attention_overlay(frame, attention_data)
        annotated_frame = self.draw_statistics(annotated_frame)
        
        cv2.imwrite(filename, annotated_frame)
        print(f"Frame saved as {filename}")
    
    def reset_statistics(self) -> None:
        """Reset session statistics."""
        self.frame_count = 0
        self.attention_scores = []
        self.attentive_frames = 0
        self.start_time = time.time()
        print("Statistics reset")
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """Process a single frame for attention detection.
        
        Args:
            frame: Input frame
            
        Returns:
            Tuple of (processed_frame, attention_data)
        """
        # Convert BGR to RGB for processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use the process_image method which handles the full pipeline
        landmarks = self.attention_detector.face_detector.process_image(rgb_frame)
        
        if landmarks and len(landmarks) > 10:  # Ensure we have enough landmarks
            attention_data = self.attention_detector.analyze_attention(landmarks)
            
            # Update statistics
            self.frame_count += 1
            if 'attention_score' in attention_data:
                self.attention_scores.append(attention_data['attention_score'])
                if attention_data['is_attentive']:
                    self.attentive_frames += 1
        else:
            # More detailed error information
            if landmarks is None:
                attention_data = {'error': 'No face detected'}
            elif len(landmarks) <= 10:
                attention_data = {'error': f'Insufficient landmarks: {len(landmarks)} found, need >10'}
            else:
                attention_data = {'error': 'Unknown detection error'}
        
        # Draw overlays
        processed_frame = self.draw_attention_overlay(frame, attention_data)
        processed_frame = self.draw_statistics(processed_frame)
        processed_frame = self.draw_controls(processed_frame)
        
        return processed_frame, attention_data
    
    def run(self, camera_index: int = 0) -> None:
        """Run the real-time attention detection system.
        
        Args:
            camera_index: Camera index to use
        """
        if not self.initialize_camera(camera_index):
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        print("\n🎥 Real-time Attention Detection Started")
        print("=" * 50)
        print("Controls:")
        print("  Q or ESC: Quit")
        print("  S: Save current frame")
        print("  R: Reset statistics")
        print("  SPACE: Pause/Resume")
        print("=" * 50)
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        
        try:
            while self.is_running:
                if not self.is_paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        print("Error: Could not read frame from camera")
                        break
                    
                    # Flip frame horizontally for mirror effect
                    frame = cv2.flip(frame, 1)
                    
                    # Process frame
                    processed_frame, attention_data = self.process_frame(frame)
                    
                    # Display frame
                    cv2.imshow(self.window_name, processed_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord('s'):  # Save frame
                    if not self.is_paused:
                        self.save_frame(frame, attention_data)
                elif key == ord('r'):  # Reset statistics
                    self.reset_statistics()
                elif key == ord(' '):  # Pause/Resume
                    self.is_paused = not self.is_paused
                    status = "PAUSED" if self.is_paused else "RESUMED"
                    print(f"Video {status}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        """Clean up resources."""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print final statistics
        if self.frame_count > 0:
            avg_score = sum(self.attention_scores) / len(self.attention_scores) if self.attention_scores else 0
            attention_percentage = (self.attentive_frames / self.frame_count) * 100
            elapsed_time = time.time() - self.start_time if self.start_time else 0
            
            print("\n📊 FINAL SESSION STATISTICS")
            print("=" * 40)
            print(f"Duration: {elapsed_time:.1f} seconds")
            print(f"Total Frames: {self.frame_count}")
            print(f"Average Attention Score: {avg_score:.2f}")
            print(f"Attentive Frames: {self.attentive_frames}/{self.frame_count} ({attention_percentage:.1f}%)")
            print("=" * 40)
        
        print("🎥 Webcam attention detection stopped")

def main():
    """Main function to run the webcam attention detector."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time Webcam Attention Detection')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index (default: 0)')
    parser.add_argument('--model', type=str, default='models/mediapipe_face.onnx',
                       help='Path to MediaPipe ONNX model')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        return
    
    try:
        detector = WebcamAttentionDetector(args.model)
        detector.run(args.camera)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 