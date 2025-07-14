#!/usr/bin/env python3
"""
Quick Launcher for Webcam Attention Detection
============================================

Simple script to launch the webcam attention detection system.

Usage:
    python run_webcam_attention.py
"""

import sys
import os

def main():
    """Launch the webcam attention detection system."""
    
    print("🎥 Launching Webcam Attention Detection System...")
    print("=" * 60)
    
    # Check if required files exist
    required_files = [
        'models/mediapipe_face.onnx',
        'webcam_attention_detector.py',
        'attention_detector.py',
        'face_landmark_detector_final.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Error: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present.")
        return
    
    print("✅ All required files found")
    print("\n📋 SYSTEM FEATURES:")
    print("   • Real-time face landmark detection")
    print("   • Multi-factor attention analysis")
    print("   • Live attention scoring")
    print("   • Session statistics tracking")
    print("   • Frame capture capability")
    
    print("\n🎮 CONTROLS:")
    print("   • Q or ESC: Quit application")
    print("   • S: Save current frame")
    print("   • R: Reset statistics")
    print("   • SPACE: Pause/Resume")
    
    print("\n🧠 ATTENTION FACTORS:")
    print("   • Eye openness (35% weight)")
    print("   • Eye symmetry (35% weight)")
    print("   • Head turning (25% weight)")
    print("   • Facial symmetry (5% weight)")
    
    print("\n🎯 ATTENTION THRESHOLD: 85%")
    print("=" * 60)
    
    # Import and run the webcam detector
    try:
        from webcam_attention_detector import WebcamAttentionDetector
        
        detector = WebcamAttentionDetector()
        detector.run()
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please ensure all required Python packages are installed.")
        print("Run: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 