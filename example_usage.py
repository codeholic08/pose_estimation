#!/usr/bin/env python3
"""
Example usage of the face landmark detector
==========================================

This script demonstrates how to use the face landmark detector
both from command line and programmatically.
"""

from face_landmark_detector_final import detect_face_landmarks
import os

def main():
    """Example usage of the face landmark detector."""
    
    # Example image path
    image_path = "A1.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found!")
        return
    
    print("Face Landmark Detection Examples")
    print("=" * 40)
    
    # Example 1: Basic landmark detection
    print("\n1. Basic landmark detection...")
    output_path = detect_face_landmarks(image_path, verbose=True)
    if output_path:
        print(f"   ✓ Success: {output_path}")
    else:
        print("   ✗ Failed")
    
    # Example 2: Face mesh visualization
    print("\n2. Face mesh visualization...")
    output_path = detect_face_landmarks(
        image_path, 
        draw_mesh=True,
        output_path="A1_example_mesh.jpg",
        verbose=True
    )
    if output_path:
        print(f"   ✓ Success: {output_path}")
    else:
        print("   ✗ Failed")
    
    # Example 3: Custom styling
    print("\n3. Custom styling (red dots, blue lines)...")
    output_path = detect_face_landmarks(
        image_path,
        draw_mesh=True,
        dot_size=3,
        dot_color=(0, 0, 255),      # Red dots (BGR format)
        line_color=(255, 0, 0),     # Blue lines (BGR format)
        line_thickness=2,
        output_path="A1_custom_style.jpg",
        verbose=True
    )
    if output_path:
        print(f"   ✓ Success: {output_path}")
    else:
        print("   ✗ Failed")
    
    print("\n" + "=" * 40)
    print("Examples completed!")
    print("\nCommand line usage:")
    print("  python face_landmark_detector_final.py --image A1.jpg")
    print("  python face_landmark_detector_final.py --image A1.jpg --mesh")


if __name__ == "__main__":
    main() 