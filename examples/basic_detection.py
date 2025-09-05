#!/usr/bin/env python3
"""
Basic Marine Species Detection Example

This script demonstrates how to use the marine detection models
for basic image processing. Perfect for beginners!

Author: Aditya Prakash
Email: aditya_2312res46@iitp.ac.in
"""

import os
import sys
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from ultralytics import YOLO
    from marine_detect.predict import predict_on_images
except ImportError as e:
    print(f"Import error: {e}")
    print("Please install required packages: pip install -r requirements.txt")
    sys.exit(1)


def basic_detection_demo():
    """
    Demonstrate basic detection on a single image.
    """
    print("🌊 Marine Species Detection Demo")
    print("=" * 40)
    
    # Paths (modify these according to your setup)
    image_path = "assets/images/input_folder/regq.jpg"
    model_path = "path/to/your/model.pt"  # Replace with actual model path
    
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        print("Please add sample images to the assets/images/input_folder/ directory")
        return
    
    # For demo purposes, we'll use a standard YOLO model
    # Replace this with your trained marine species model
    try:
        print("📥 Loading model...")
        model = YOLO('yolov8n.pt')  # This will download automatically
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Load and display original image
    print("🖼️  Processing image...")
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run detection
    results = model(image_path, conf=0.3)
    
    # Display results
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    plt.axis('off')
    
    # Detection results
    plt.subplot(1, 2, 2)
    annotated = results[0].plot()
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    plt.imshow(annotated_rgb)
    plt.title("Detection Results")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detection statistics
    boxes = results[0].boxes
    if boxes is not None:
        print(f"📊 Found {len(boxes)} detections:")
        for i, box in enumerate(boxes):
            class_id = int(box.cls)
            confidence = float(box.conf)
            class_name = model.names[class_id]
            print(f"  {i+1}. {class_name}: {confidence:.2f} confidence")
    else:
        print("❌ No detections found. Try lowering the confidence threshold.")


def batch_processing_demo():
    """
    Demonstrate batch processing of multiple images.
    """
    print("\n🗂️  Batch Processing Demo")
    print("=" * 40)
    
    # Define paths
    input_folder = "assets/images/input_folder/"
    output_folder = "assets/images/output_folder/"
    
    # Check if input folder exists
    if not os.path.exists(input_folder):
        print(f"❌ Input folder not found: {input_folder}")
        return
    
    # List images in input folder
    image_files = [f for f in os.listdir(input_folder) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("❌ No image files found in input folder")
        return
    
    print(f"📁 Found {len(image_files)} images to process")
    
    # Example of how to use the batch processing function
    # (You'll need trained models for this to work)
    model_paths = [
        "path/to/your/fish_model.pt",
        "path/to/your/megafauna_model.pt"
    ]
    conf_thresholds = [0.5, 0.6]
    
    print("⚠️  Note: Replace model_paths with your actual trained models")
    print("Example usage:")
    print(f"predict_on_images(")
    print(f"    model_paths={model_paths},")
    print(f"    confs_threshold={conf_thresholds},")
    print(f"    images_input_folder_path='{input_folder}',")
    print(f"    images_output_folder_path='{output_folder}'")
    print(f")")


def parameter_tuning_demo():
    """
    Demonstrate the effect of different parameters.
    """
    print("\n⚙️  Parameter Tuning Demo")
    print("=" * 40)
    
    image_path = "assets/images/input_folder/regq.jpg"
    
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load model
    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Test different confidence thresholds
    confidence_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    print("🎯 Testing different confidence thresholds:")
    for conf in confidence_levels:
        results = model(image_path, conf=conf)
        num_detections = len(results[0].boxes) if results[0].boxes is not None else 0
        print(f"  Confidence {conf}: {num_detections} detections")
    
    print("\n💡 Tips:")
    print("- Lower confidence = more detections (but more false positives)")
    print("- Higher confidence = fewer detections (but more accurate)")
    print("- Optimal threshold depends on your specific use case")


def main():
    """
    Main function to run all demos.
    """
    print("🐟 Marine Species Detection - Student Examples")
    print("=" * 50)
    print("Author: Aditya Prakash")
    print("Email: aditya_2312res46@iitp.ac.in")
    print("=" * 50)
    
    try:
        # Run demos
        basic_detection_demo()
        batch_processing_demo()
        parameter_tuning_demo()
        
        print("\n🎓 Learning Tips:")
        print("1. Start with the basic detection demo")
        print("2. Experiment with different confidence thresholds")
        print("3. Try your own images")
        print("4. Train models on marine species data")
        print("5. Implement your own evaluation metrics")
        
        print("\n📚 Next Steps:")
        print("- Check out the tutorial.ipynb for interactive learning")
        print("- Read the README.md for detailed documentation")
        print("- Explore the learning resources section")
        
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your setup and try again")


if __name__ == "__main__":
    main()
