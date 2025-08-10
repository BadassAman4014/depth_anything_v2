#!/usr/bin/env python3
"""
Simple launcher for the Wound Segmentation Gradio App
"""

import sys
import os

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = ['gradio', 'tensorflow', 'cv2', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed!")
    return True

def check_model_files():
    """Check if model files exist"""
    model_files = [
        'training_history/2025-08-07_12-30-43.hdf5',
        'training_history/2019-12-19 01%3A53%3A15.480800.hdf5'
    ]
    
    existing_models = []
    for model_file in model_files:
        if os.path.exists(model_file):
            existing_models.append(model_file)
    
    if not existing_models:
        print("❌ No model files found!")
        print("   Please ensure you have trained models in the training_history/ directory")
        return False
    
    print(f"✅ Found {len(existing_models)} model file(s):")
    for model in existing_models:
        print(f"   - {model}")
    return True

def main():
    """Main function to launch the Gradio app"""
    print("🚀 Starting Wound Segmentation Gradio App...")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check model files
    if not check_model_files():
        sys.exit(1)
    
    print("\n🎯 Launching Gradio interface...")
    print("   The app will be available at: http://localhost:7860")
    print("   Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        # Import and run the Gradio app
        from gradio_app import create_gradio_interface
        
        interface = create_gradio_interface()
        interface.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=True,
            show_error=True
        )
    except KeyboardInterrupt:
        print("\n👋 Gradio app stopped by user")
    except Exception as e:
        print(f"\n❌ Error launching Gradio app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 