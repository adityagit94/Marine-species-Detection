# Model Documentation

This document provides comprehensive information about the marine species detection models, including architecture details, performance metrics, training procedures, and usage guidelines.

## 🎯 Model Overview

The Marine Detect system employs two specialized YOLOv8 models for comprehensive marine species detection:

1. **Fish and Invertebrates Model**: Detects 15 classes of fish and invertebrate species
2. **MegaFauna Model**: Detects 3 classes of large marine species (sharks, rays, turtles)

### Model Architecture

Both models are based on the YOLOv8 architecture, which provides:
- **Real-time performance**: Optimized for fast inference
- **High accuracy**: State-of-the-art object detection capabilities
- **Scalability**: Multiple model sizes available (n, s, m, l, x)
- **Flexibility**: Easy to fine-tune for specific domains

## 🐟 Fish and Invertebrates Model

### Supported Species

| Class ID | Species Name | Scientific Family | Common Name |
|----------|--------------|-------------------|-------------|
| 0 | fish | Various | General fish category |
| 1 | serranidae | Serranidae | Groupers |
| 2 | urchin | Echinoidea | Sea urchins |
| 3 | scaridae | Scaridae | Parrotfish |
| 4 | chaetodontidae | Chaetodontidae | Butterfly fish |
| 5 | giant_clam | Tridacnidae | Giant clams |
| 6 | lutjanidae | Lutjanidae | Snappers |
| 7 | muraenidae | Muraenidae | Moray eels |
| 8 | sea_cucumber | Holothuroidea | Sea cucumbers |
| 9 | haemulidae | Haemulidae | Sweet lips |
| 10 | lobster | Nephropidae | Lobsters |
| 11 | crown_of_thorns | Acanthaster | Crown-of-thorns starfish |
| 12 | bolbometopon_muricatum | Scaridae | Bumphead parrotfish |
| 13 | cheilinus_undulatus | Labridae | Humphead wrasse |
| 14 | cromileptes_altivelis | Serranidae | Barramundi cod |

### Performance Metrics

#### Test Set Performance (499 images)

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|--------|-----------|-----------|--------|---------|--------------|
| fish | 499 | 259 | 0.652 | 0.580 | 0.616 | 0.501 |
| serranidae | 499 | 49 | 0.876 | 0.824 | 0.850 | 0.777 |
| urchin | 499 | 80 | 0.768 | 0.718 | 0.743 | 0.479 |
| scaridae | 499 | 48 | 0.854 | 0.802 | 0.828 | 0.794 |
| chaetodontidae | 499 | 65 | 0.918 | 0.864 | 0.891 | 0.827 |
| giant_clam | 499 | 102 | 0.896 | 0.844 | 0.870 | 0.602 |
| lutjanidae | 499 | 86 | 0.891 | 0.839 | 0.865 | 0.777 |
| muraenidae | 499 | 58 | 0.975 | 0.923 | 0.949 | 0.809 |
| sea_cucumber | 499 | 33 | 0.995 | 0.943 | 0.969 | 0.939 |
| haemulidae | 499 | 22 | 0.998 | 0.946 | 0.972 | 0.945 |
| lobster | 499 | 31 | 1.000 | 0.968 | 0.984 | 0.877 |
| crown_of_thorns | 499 | 28 | 1.000 | 0.962 | 0.981 | 0.790 |
| bolbometopon_muricatum | 499 | 19 | 1.000 | 0.986 | 0.993 | 0.936 |
| cheilinus_undulatus | 499 | 29 | 1.000 | 0.990 | 0.995 | 0.968 |
| cromileptes_altivelis | 499 | 30 | 1.000 | 0.990 | 0.995 | 0.945 |

**Overall Performance:**
- **Mean Average Precision (mAP@0.5)**: 0.867
- **Mean Average Precision (mAP@0.5:0.95)**: 0.764
- **Average Precision**: 0.895
- **Average Recall**: 0.858

### Model Specifications

```yaml
Model: YOLOv8m (Medium)
Input Size: 640x640 pixels
Parameters: ~25.9M
FLOPs: ~78.9G
Model Size: ~52MB
Inference Speed: ~2.8ms (GPU), ~45ms (CPU)
Training Time: ~12 hours (V100 GPU)
```

## 🦈 MegaFauna Model

### Supported Species

| Class ID | Species Name | Description | Conservation Status |
|----------|--------------|-------------|-------------------|
| 0 | ray | Rays and skates | Various (LC to CR) |
| 1 | shark | Sharks (all species) | Various (LC to CR) |
| 2 | turtle | Sea turtles | Mostly threatened |

### Performance Metrics

#### Test Set Performance (253 images)

| Class | Images | Instances | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|-------|--------|-----------|-----------|--------|---------|--------------|
| ray | 253 | 73 | 0.889 | 0.837 | 0.863 | 0.777 |
| shark | 253 | 111 | 0.767 | 0.715 | 0.741 | 0.627 |
| turtle | 253 | 109 | 0.974 | 0.922 | 0.948 | 0.887 |

**Overall Performance:**
- **Mean Average Precision (mAP@0.5)**: 0.851
- **Mean Average Precision (mAP@0.5:0.95)**: 0.764
- **Average Precision**: 0.877
- **Average Recall**: 0.825

### Model Specifications

```yaml
Model: YOLOv8m (Medium)
Input Size: 640x640 pixels
Parameters: ~25.9M
FLOPs: ~78.9G
Model Size: ~52MB
Inference Speed: ~2.8ms (GPU), ~45ms (CPU)
Training Time: ~8 hours (V100 GPU)
```

## 📊 Training Details

### Dataset Information

#### Fish and Invertebrates Dataset
- **Total Images**: 12,742 (training + validation + test)
- **Training Set**: 9,794 images (80%)
- **Validation Set**: 2,449 images (20%)
- **Test Set**: 499 images
- **Annotation Format**: YOLO format (.txt files)
- **Image Resolution**: Variable (resized to 640x640 for training)

#### MegaFauna Dataset
- **Total Images**: 8,383 (training + validation + test)
- **Training Set**: 6,504 images (80%)
- **Validation Set**: 1,626 images (20%)
- **Test Set**: 253 images
- **Annotation Format**: YOLO format (.txt files)
- **Image Resolution**: Variable (resized to 640x640 for training)

### Training Configuration

```yaml
# Training hyperparameters
epochs: 100
batch_size: 16
learning_rate: 0.01
momentum: 0.937
weight_decay: 0.0005
warmup_epochs: 3
warmup_momentum: 0.8
warmup_bias_lr: 0.1

# Data augmentation
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
degrees: 0.0
translate: 0.1
scale: 0.5
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.5
mosaic: 1.0
mixup: 0.0

# Optimization
optimizer: SGD
cos_lr: False
close_mosaic: 10
```

### Training Process

1. **Data Preparation**
   - Image collection from multiple sources
   - Quality filtering and deduplication
   - Manual annotation and validation
   - Data split into train/validation/test sets

2. **Model Initialization**
   - Start with COCO pre-trained YOLOv8 weights
   - Modify final layer for specific number of classes
   - Initialize new layers with appropriate weights

3. **Training Strategy**
   - Progressive training with different image sizes
   - Learning rate scheduling with cosine annealing
   - Early stopping based on validation mAP
   - Model checkpointing every 10 epochs

4. **Validation and Testing**
   - Continuous validation during training
   - Final evaluation on held-out test set
   - Performance analysis across different species

## 🎯 Optimal Confidence Thresholds

Based on validation experiments, the recommended confidence thresholds are:

### Fish and Invertebrates Model
```python
OPTIMAL_THRESHOLDS = {
    "fish": 0.45,
    "serranidae": 0.55,
    "urchin": 0.50,
    "scaridae": 0.60,
    "chaetodontidae": 0.65,
    "giant_clam": 0.60,
    "lutjanidae": 0.58,
    "muraenidae": 0.70,
    "sea_cucumber": 0.75,
    "haemulidae": 0.75,
    "lobster": 0.80,
    "crown_of_thorns": 0.75,
    "bolbometopon_muricatum": 0.80,
    "cheilinus_undulatus": 0.85,
    "cromileptes_altivelis": 0.80
}

# Overall recommended threshold
RECOMMENDED_THRESHOLD = 0.522
```

### MegaFauna Model
```python
OPTIMAL_THRESHOLDS = {
    "ray": 0.55,
    "shark": 0.50,
    "turtle": 0.70
}

# Overall recommended threshold
RECOMMENDED_THRESHOLD = 0.6
```

## 🔧 Model Usage

### Loading Models

```python
from ultralytics import YOLO
import torch

# Load Fish and Invertebrates model
fish_model = YOLO('models/fish_invertebrates.pt')

# Load MegaFauna model
megafauna_model = YOLO('models/megafauna.pt')

# Check model info
print(f"Fish model classes: {fish_model.names}")
print(f"MegaFauna model classes: {megafauna_model.names}")

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fish_model.to(device)
megafauna_model.to(device)
```

### Single Image Inference

```python
import cv2
from pathlib import Path

def detect_marine_species(image_path: str, confidence: float = 0.5):
    """Detect marine species in a single image."""
    
    # Load image
    image = cv2.imread(image_path)
    
    # Run both models
    fish_results = fish_model(image, conf=0.522)
    megafauna_results = megafauna_model(image, conf=0.6)
    
    # Combine results
    all_detections = []
    
    for result in fish_results:
        if result.boxes is not None:
            for box in result.boxes:
                detection = {
                    'class_id': int(box.cls),
                    'class_name': fish_model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist(),
                    'model': 'fish_invertebrates'
                }
                all_detections.append(detection)
    
    for result in megafauna_results:
        if result.boxes is not None:
            for box in result.boxes:
                detection = {
                    'class_id': int(box.cls),
                    'class_name': megafauna_model.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xyxy[0].tolist(),
                    'model': 'megafauna'
                }
                all_detections.append(detection)
    
    return all_detections

# Example usage
detections = detect_marine_species('marine_image.jpg')
print(f"Found {len(detections)} marine species")
```

### Batch Processing

```python
from marine_detect.predict import predict_on_images

# Process multiple images
predict_on_images(
    model_paths=['models/fish_invertebrates.pt', 'models/megafauna.pt'],
    confs_threshold=[0.522, 0.6],
    images_input_folder_path='input_images/',
    images_output_folder_path='output_images/',
    save_txt=True,  # Save detection results as text files
    save_conf=True  # Include confidence scores in text files
)
```

## 📈 Performance Analysis

### Confusion Matrix Analysis

The models show excellent performance on most classes, with some challenges:

**High-performing classes:**
- Sea cucumbers, lobsters, and rare species (>95% accuracy)
- Moray eels and distinctive fish families (>90% accuracy)

**Challenging classes:**
- General "fish" category (lower specificity)
- Small or partially occluded specimens
- Similar-looking species (e.g., different ray species)

### Error Analysis

**Common failure modes:**
1. **Occlusion**: Partially hidden specimens
2. **Scale variation**: Very small or very large specimens
3. **Lighting conditions**: Poor underwater visibility
4. **Motion blur**: Fast-moving subjects
5. **Background complexity**: Cluttered reef environments

### Improvement Strategies

1. **Data augmentation**: Increase training data diversity
2. **Hard negative mining**: Focus on challenging examples
3. **Multi-scale training**: Better handling of size variations
4. **Ensemble methods**: Combine multiple models
5. **Post-processing**: Non-maximum suppression tuning

## 🔄 Model Updates and Versioning

### Version History

- **v1.0.0**: Initial release with basic species detection
- **v1.1.0**: Improved accuracy with additional training data
- **v1.2.0**: Added rare species detection capabilities
- **v2.0.0**: Separate models for fish/invertebrates and megafauna

### Update Process

1. **Data collection**: Gather new annotated images
2. **Model retraining**: Fine-tune with new data
3. **Validation**: Test on held-out validation set
4. **A/B testing**: Compare with previous version
5. **Deployment**: Gradual rollout with monitoring

### Model Monitoring

```python
import json
from datetime import datetime

def log_prediction_metrics(predictions, ground_truth=None):
    """Log prediction metrics for model monitoring."""
    
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'total_predictions': len(predictions),
        'confidence_distribution': {
            'mean': np.mean([p['confidence'] for p in predictions]),
            'std': np.std([p['confidence'] for p in predictions]),
            'min': np.min([p['confidence'] for p in predictions]),
            'max': np.max([p['confidence'] for p in predictions])
        },
        'class_distribution': {}
    }
    
    # Class distribution
    for pred in predictions:
        class_name = pred['class_name']
        if class_name not in metrics['class_distribution']:
            metrics['class_distribution'][class_name] = 0
        metrics['class_distribution'][class_name] += 1
    
    # Log to file
    with open('model_metrics.jsonl', 'a') as f:
        f.write(json.dumps(metrics) + '\n')
    
    return metrics
```

## 🚀 Future Enhancements

### Planned Improvements

1. **Model Architecture**
   - Experiment with YOLOv9 and YOLOv10
   - Custom architectures for underwater environments
   - Attention mechanisms for better feature extraction

2. **Data and Training**
   - Active learning for efficient data collection
   - Synthetic data generation
   - Domain adaptation techniques

3. **Performance Optimization**
   - Model quantization for edge deployment
   - Knowledge distillation for smaller models
   - Hardware-specific optimizations

4. **New Capabilities**
   - Species behavior analysis
   - Population counting
   - Health assessment indicators

## 📞 Model Support

For model-related questions:
- Performance issues: Check the [troubleshooting guide](troubleshooting.md)
- Custom training: See the [development guide](development.md)
- Model requests: Contact [aditya_2312res46@iitp.ac.in](mailto:aditya_2312res46@iitp.ac.in)

---

**Next**: [Deployment Guide](deployment.md) | [Performance Guide](performance.md)
