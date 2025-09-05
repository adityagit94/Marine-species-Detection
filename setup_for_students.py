#!/usr/bin/env python3
"""
Student Setup Script for Marine Detection Project

This script helps students set up their environment and check
that everything is working correctly.

Author: Aditya Prakash
Email: aditya_2312res46@iitp.ac.in
"""

import os
import sys
import subprocess
import importlib.util
from pathlib import Path


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False


def check_required_packages():
    """Check if required packages are installed."""
    print("\n📦 Checking required packages...")
    
    required_packages = [
        'cv2',
        'numpy',
        'PIL',
        'matplotlib',
        'ultralytics',
        'tqdm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✅ OpenCV {cv2.__version__}")
            elif package == 'PIL':
                from PIL import Image
                print(f"✅ Pillow (PIL)")
            elif package == 'ultralytics':
                import ultralytics
                print(f"✅ Ultralytics {ultralytics.__version__}")
            else:
                spec = importlib.util.find_spec(package)
                if spec is not None:
                    module = importlib.import_module(package)
                    version = getattr(module, '__version__', 'unknown')
                    print(f"✅ {package} {version}")
                else:
                    raise ImportError
        except ImportError:
            print(f"❌ {package} - Not installed")
            missing_packages.append(package)
    
    return missing_packages


def install_missing_packages(missing_packages):
    """Install missing packages."""
    if not missing_packages:
        return True
    
    print(f"\n🔧 Installing missing packages: {', '.join(missing_packages)}")
    
    # Map package names to pip install names
    pip_names = {
        'cv2': 'opencv-python',
        'PIL': 'Pillow'
    }
    
    for package in missing_packages:
        pip_name = pip_names.get(package, package)
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pip_name])
            print(f"✅ Installed {pip_name}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {pip_name}")
            return False
    
    return True


def create_directory_structure():
    """Create necessary directories if they don't exist."""
    print("\n📁 Setting up directory structure...")
    
    directories = [
        'assets/images/input_folder',
        'assets/images/output_folder',
        'models',
        'examples',
        'data'
    ]
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created directory: {directory}")
        else:
            print(f"📁 Directory exists: {directory}")


def download_sample_model():
    """Download a sample YOLO model for testing."""
    print("\n🤖 Setting up sample model...")
    
    try:
        from ultralytics import YOLO
        # This will download YOLOv8n automatically
        model = YOLO('yolov8n.pt')
        print("✅ Sample YOLOv8n model downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed to download sample model: {e}")
        return False


def create_sample_config():
    """Create a sample configuration file."""
    print("\n⚙️ Creating sample configuration...")
    
    config_content = """# Marine Detection Configuration
# Modify these paths according to your setup

# Model paths (replace with your trained models)
FISH_MODEL_PATH = "models/fish_model.pt"
MEGAFAUNA_MODEL_PATH = "models/megafauna_model.pt"

# Confidence thresholds
FISH_CONFIDENCE = 0.5
MEGAFAUNA_CONFIDENCE = 0.6

# Input/Output paths
INPUT_FOLDER = "assets/images/input_folder/"
OUTPUT_FOLDER = "assets/images/output_folder/"

# Processing settings
BATCH_SIZE = 1
DEVICE = "cpu"  # Change to "cuda" if you have GPU
IMAGE_SIZE = 640
"""
    
    config_path = Path("config.py")
    if not config_path.exists():
        with open(config_path, 'w') as f:
            f.write(config_content)
        print("✅ Created config.py")
    else:
        print("📄 config.py already exists")


def run_basic_test():
    """Run a basic functionality test."""
    print("\n🧪 Running basic functionality test...")
    
    try:
        from ultralytics import YOLO
        import cv2
        import numpy as np
        
        # Create a dummy image
        dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Load model and run inference
        model = YOLO('yolov8n.pt')
        results = model(dummy_image)
        
        print("✅ Basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def print_next_steps():
    """Print next steps for students."""
    print("\n🎓 Setup Complete! Next Steps:")
    print("=" * 50)
    print("1. 📖 Read the README.md for detailed documentation")
    print("2. 📓 Open tutorial.ipynb in Jupyter for interactive learning")
    print("3. 🏃 Run examples/basic_detection.py to see a demo")
    print("4. 🖼️  Add your own images to assets/images/input_folder/")
    print("5. 🤖 Train or download marine species detection models")
    print()
    print("📚 Learning Resources:")
    print("- Tutorial notebook: tutorial.ipynb")
    print("- Example scripts: examples/")
    print("- Documentation: README.md")
    print("- Configuration: config.py")
    print()
    print("❓ Need Help?")
    print("- Check the troubleshooting section in README.md")
    print("- Review the FAQ section")
    print("- Experiment with the provided examples")
    print()
    print("🌊 Happy coding! 🐟")


def main():
    """Main setup function."""
    print("🌊 Marine Species Detection - Student Setup")
    print("=" * 50)
    print("Author: Aditya Prakash")
    print("Email: aditya_2312res46@iitp.ac.in")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup failed: Incompatible Python version")
        return False
    
    # Check and install packages
    missing = check_required_packages()
    if missing:
        install_choice = input(f"\nInstall missing packages? (y/n): ").lower().strip()
        if install_choice == 'y':
            if not install_missing_packages(missing):
                print("\n❌ Setup failed: Could not install required packages")
                return False
        else:
            print("\n⚠️  Setup incomplete: Missing required packages")
            return False
    
    # Create directories
    create_directory_structure()
    
    # Download sample model
    download_sample_model()
    
    # Create configuration
    create_sample_config()
    
    # Run basic test
    if not run_basic_test():
        print("\n⚠️  Setup completed with warnings")
    else:
        print("\n✅ Setup completed successfully!")
    
    # Print next steps
    print_next_steps()
    
    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Setup failed with error: {e}")
        print("Please check your environment and try again")
