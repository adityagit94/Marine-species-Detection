# Marine Detect 🌊🐟

This repository provides access to two **YOLOv8 object detection models** for identifying **species** of interest in **underwater environments**.

<p align="center">
<img src="./assets/gif/results.gif" height="250"/>
<img src="./assets/gif/turtle+fish-resized.gif" height="250"/>
</p>

These models were developed for marine conservation and research purposes, utilizing advanced computer vision techniques to automate the identification and quantification of marine species. The system is designed to assist marine biologists and researchers in monitoring underwater ecosystems by automatically detecting and classifying various marine species from images and video footage.

## 🎓 For Students

This project is an excellent learning resource for computer science students interested in:
- **Computer Vision**: Understanding object detection with YOLO
- **Machine Learning**: Training and deploying deep learning models
- **Environmental AI**: Applying AI for conservation and environmental monitoring
- **Python Programming**: Working with OpenCV, PyTorch, and data processing
- **Research Methods**: Implementing and evaluating ML models for real-world applications

### 🚀 Quick Start for Students

1. **Setup Environment**: Run the setup script to get started quickly
   ```bash
   python setup_for_students.py
   ```

2. **Interactive Learning**: Open the tutorial notebook
   ```bash
   jupyter notebook tutorial.ipynb
   ```

3. **Run Examples**: Try the basic detection example
   ```bash
   python examples/basic_detection.py
   ```

4. **Explore and Learn**: Check out the learning exercises and resources below!

## 📚 Documentation

This project includes comprehensive documentation to help you get started and contribute effectively:

### 📖 User Documentation
- **[Installation Guide](docs/installation.md)** - Complete setup instructions for all environments
- **[API Documentation](docs/api.md)** - REST API reference with examples
- **[Model Documentation](docs/models.md)** - Detailed model information and performance metrics
- **[Troubleshooting Guide](docs/troubleshooting.md)** - Common issues and solutions

### 🔧 Developer Documentation  
- **[Development Guide](docs/development.md)** - Contributing guidelines and development setup
- **[Architecture Overview](docs/architecture.md)** - System design and component interactions
- **[Deployment Guide](docs/deployment.md)** - Production deployment instructions
- **[Performance Guide](docs/performance.md)** - Optimization tips and benchmarking

### 📝 Project Information
- **[Changelog](CHANGELOG.md)** - Version history and release notes
- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute to the project
- **[License](LICENSE)** - Project license information

## 🐟 Species Scope

The *Fish and Invertebrates* Object Detection Model detects the *Fish and Invertebrates Species* and the *MegaFauna* Object Detection Model detects *MegaFauna and Rare Species*.

- **MegaFauna and Rare Species**: Sharks, Sea Turtles, Rays.
- **Fish Species**: Butterfly Fish (Chaetodontidae), Grouper (Serranidae), Parrotfish (Scaridae), Snapper (Lutjanidae), Moray Eel (Muraenidae), Sweet Lips (Haemulidae), Barramundi Cod (Cromileptes altivelis), Humphead (Napoleon) Wrasse (Cheilinus undulatus), Bumphead Parrotfish (Bolbometopon muricatum), Fish (other than above or unrecognizable).
- **Invertebrates Species**: Giant Clam, Urchin, Sea Cucumber, Lobster, Crown of Thorns.

These species are **"bio-indicating"** species, which serve as indicators of the ecosystem health. These bio-indicating species are of course dependent on each region - here the focus is for Malaysia/Indo-Pacific region.

## 📊 Datasets Details

The models utilize a combination of publicly available datasets and custom annotated datasets. Some datasets were already annotated, and others underwent manual labeling to ensure high-quality training data.

References to the public datasets used can be found in the 'References' section of this README.

### Datasets split details

| Model          | Training + Validation Sets | Test Set     |
| -------------- | -------------------------- | --------     |
| FishInv        | 12,243 images (80%, 20%)   | 499  images  |
| MegaFauna      | 8,130 images (80%, 20%)    | 253  images  |

> [!NOTE]
> The rationale behind the development of two distinct models lies in the utilization of already annotated images available in public datasets. By having separate models, we sidestep the necessity of reannotating images that already encompass annotations for specific species with every Fish, Invertebrates and MegaFauna species.  For example, we found a lot of images of turtles already annotated. If we were to adopt a single, all-encompassing model for both Fish and Invertebrates Species 🐟 and MegaFauna 🦈, it would necessitate the reannotation of all those turtle images to include species like urchins, fishes, ...

## 🤖 Model Details

The trained models are available for download. Contact the repository maintainer for access to the pre-trained model files:

### Performances on test sets

> [!IMPORTANT]
> Our models are currently undergoing enhancements for improved performance. More labeled images are on the way and will be used to retrain the models.

<details>

<summary>MegaFauna model performances</summary>

| Class  | Images | Instances | mAP50 | mAP50-95 |
| ------ | ------ | --------- | ----- | -------- |
| ray    | 253    | 73        | 0.863 | 0.777    |
| shark  | 253    | 111       | 0.741 | 0.627    |
| turtle | 253    | 109       | 0.948 | 0.887    |

</details>

<details>

<summary>FishInv model performances</summary>

| Class                  | Images | Instances | mAP50 | mAP50-95 |
| ---------------------- | ------ | --------- | ----- | -------- |
| fish                   | 499    | 259       | 0.616 | 0.501    |
| serranidae             | 499    | 49        | 0.850 | 0.777    |
| urchin                 | 499    | 80        | 0.743 | 0.479    |
| scaridae               | 499    | 48        | 0.828 | 0.794    |
| chaetodontidae         | 499    | 65        | 0.891 | 0.827    |
| giant_clam             | 499    | 102       | 0.870 | 0.602    |
| lutjanidae             | 499    | 86        | 0.865 | 0.777    |
| muraenidae             | 499    | 58        | 0.949 | 0.809    |
| sea_cucumber           | 499    | 33        | 0.969 | 0.939    |
| haemulidae             | 499    | 22        | 0.972 | 0.945    |
| lobster                | 499    | 31        | 0.984 | 0.877    |
| crown_of_thorns        | 499    | 28        | 0.981 | 0.790    |
| bolbometopon_muricatum | 499    | 19        | 0.993 | 0.936    |
| cheilinus_undulatus    | 499    | 29        | 0.995 | 0.968    |
| cromileptes_altivelis  | 499    | 30        | 0.995 | 0.945    |

</details>

## 📁 Project Structure

```
marine-detect/
├── assets/                     # Media assets
│   ├── gif/                   # Demo GIFs
│   └── images/                # Sample input/output images
├── src/                       # Source code
│   └── marine_detect/         # Main package
│       ├── __init__.py        # Package initialization
│       └── predict.py         # Inference functions
├── requirements.txt           # Python dependencies
├── pyproject.toml            # Project configuration
├── CITATION.cff              # Citation information
├── LICENSE                   # License file
└── README.md                 # This file
```

## 🚗 Usage

### 🏁 Environment Setup

There are 3 options to install the development environment.

#### Option 1 - Developing Inside a Docker Container with Visual Studio Code's Dev Containers Extension (recommended):

- If you are using Windows, make sure that Windows Subsytem for Linux is installed and working on your machine (to do so, follow the instructions [here](https://learn.microsoft.com/en-us/windows/wsl/install)).
- Make sure Docker is installed on your machine.
- Install the Dev Containers Extension in Visual Studio Code (*ms-vscode-remote.remote-containers*).
- In VS Code, open the command palette (CTRL + SHIFT + P) and select *Dev Containers: Rebuild and Reopen in Container* (make sure Docker is running before executing this step). If the build seems to freeze, read the "Common Errors and Fixes" section below.

Note that the Dockerfile was created for CPU machines. If you wish to use GPU for inference, you can change the base image to `nvidia/cuda:12.0.0-runtime-ubuntu22.04`.

#### Option 2 - Developing on Your Host OS with Anaconda:

- Make sure Conda is installed and working on your machine (to do so, click [here](https://www.anaconda.com/download)).
- Then, run the following commands in the project directory:
```shell
conda create --name your_env_name python=3.10
conda activate your_env_name
pip install -r requirements.txt
```

#### Option 3 - Developing on Your Host OS with PIP:

- Make sure pyenv is installed and working
- Then, run the following commands in the project directory:
```shell
pyenv install 3.10
pyenv local 3.10
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 🏋️ Model Training

To train your own marine species detection models:

1. **Prepare Dataset**: Organize your annotated marine species images in YOLO format
2. **Train Models**: Use YOLOv8 to train custom models on your dataset
3. **Validation**: Evaluate model performance on test sets

```python
from ultralytics import YOLO

# Load a YOLOv8 model
model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt

# Train the model
results = model.train(data='path/to/your/dataset.yaml', epochs=100, imgsz=640)
```

### 🚀 Inference

To make predictions on images or videos using the marine detection models, follow these simple steps:

1. **Models Setup**: Download or train your own YOLOv8 models for marine species detection and place them in an accessible directory.
2. **Prediction Functions**: Utilize the following Python functions to generate predictions with bounding box annotations.

```python
from src.marine_detect.predict import predict_on_images, predict_on_video

# Predict on a set of images using your trained models
predict_on_images(
    model_paths=["path/to/fish_model.pt", "path/to/megafauna_model.pt"],
    confs_threshold=[0.5, 0.6],  # Adjust confidence thresholds as needed
    images_input_folder_path="path/to/input/images",
    images_output_folder_path="path/to/output/folder",
)

# Predict on a video using your trained models
predict_on_video(
    model_paths=["path/to/fish_model.pt", "path/to/megafauna_model.pt"],
    confs_threshold=[0.5, 0.6],  # Adjust confidence thresholds as needed
    input_video_path="path/to/input/video.mp4",
    output_video_path="path/to/output/video.mp4",
)
```
> [!NOTE]
> Adjust the confidence thresholds based on your specific models and requirements. Higher values result in more confident detections but may miss some species.

The resulting images or video files will have bounding boxes annotations, visually indicating the location and extent of the detected marine species within the original data. 

For example:

<p align="middle">
  <img src="./assets/images/input_folder/regq.jpg" width="300" />
  <img src="./assets/images/output_folder/regq.jpg" width="300" /> 
</p>

## 🔧 Troubleshooting & FAQ

### Common Issues

**Q: ModuleNotFoundError when importing marine_detect**
```bash
# Solution: Install in development mode
pip install -e .
```

**Q: CUDA out of memory errors**
```python
# Solution: Reduce batch size or use CPU
model = YOLO('model.pt')
results = model(image, device='cpu')  # Force CPU usage
```

**Q: Poor detection performance**
- Check image quality and lighting conditions
- Adjust confidence thresholds (lower for more detections)
- Ensure your species are in the trained classes
- Consider retraining with your specific data

**Q: Video processing is slow**
- Use smaller input resolution
- Process every nth frame instead of all frames
- Use GPU acceleration if available

### Performance Tips

1. **Batch Processing**: Process multiple images at once for better efficiency
2. **Model Selection**: Use smaller models (yolov8n) for faster inference, larger (yolov8x) for better accuracy
3. **Image Preprocessing**: Resize images to optimal resolution (640x640) before inference
4. **Hardware**: Use GPU for training and inference when possible

## 🎯 Learning Exercises for Students

### Beginner Level
1. **Setup and Run**: Get the environment working and run inference on sample images
2. **Parameter Tuning**: Experiment with different confidence thresholds
3. **Visualization**: Modify the plotting functions to change bounding box colors/styles

### Intermediate Level
1. **Data Analysis**: Analyze the model performance metrics and create visualizations
2. **Custom Dataset**: Collect and annotate your own marine species images
3. **Model Comparison**: Compare different YOLO model sizes (n, s, m, l, x)

### Advanced Level
1. **Transfer Learning**: Fine-tune the models on new species or environments
2. **Model Optimization**: Implement model quantization or pruning for mobile deployment
3. **Real-time Processing**: Create a live video stream processing application
4. **Performance Analysis**: Implement detailed evaluation metrics and confusion matrices

## 📖 Learning Resources

### Essential Background Knowledge
- **Computer Vision Basics**: [CS231n Stanford Course](http://cs231n.stanford.edu/)
- **Deep Learning Fundamentals**: [Deep Learning Book](https://www.deeplearningbook.org/)
- **YOLO Architecture**: [You Only Look Once Paper](https://arxiv.org/abs/1506.02640)
- **Object Detection Survey**: [Object Detection with Deep Learning](https://arxiv.org/abs/1807.05511)

### Practical Tutorials
- **YOLOv8 Documentation**: [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- **OpenCV Python Tutorials**: [OpenCV-Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)
- **Marine Biology Context**: [Marine Species Classification](https://www.marinespecies.org/)

### Tools and Libraries
- **Annotation Tools**: [LabelImg](https://github.com/tzutalin/labelImg), [CVAT](https://github.com/openvinotoolkit/cvat)
- **Model Visualization**: [Netron](https://netron.app/), [TensorBoard](https://www.tensorflow.org/tensorboard)
- **Data Augmentation**: [Albumentations](https://albumentations.ai/)

### Related Projects
- **Wildlife Detection**: [MegaDetector](https://github.com/microsoft/CameraTraps)
- **Marine Conservation**: [Wildbook](https://www.wildbook.org/)
- **Underwater Computer Vision**: [Underwater Image Enhancement](https://github.com/saeed-anwar/UWCNN)

## 📚 References

### Datasets

- Ticon Dataset. (2023). Shark Dataset [ Open Source Dataset ]. In Roboflow Universe . Roboflow . https://universe.roboflow.com/ticon-dataset/shark-ibmby
- Minhajul Arefin. (2021).  zebra_shark Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/minhajul-arefin/zebra_shark
- Rizal Fadia Al Fikri. (2022).  shark_species Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/rizal-fadia-al-fikri/shark_species
- Aya Abd-Elnaser. (2022).  SHARK Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/aya-abd-elnaser/shark-jatfb
- Nomi. (2023).  seaturtle Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/nomi/seaturtle
- Parvej Hosen. (2022).  Turtle Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/parvej-hosen/turtle-f9xgw
- Seami New 5 Fishes. (2023).  EagleRay New Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/seami-new-5-fishes/eagleray-new
- Le Wagon. (2023).  count-a-manta Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/le-wagon-w02yl/count-a-manta
- Renaldo Rasfuldi. (2022).  fish_id_2 Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/renaldo-rasfuldi/fish_id_2
- Universiti Teknologi Malaysia. (2023).  Giant Clam Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/universiti-teknologi-malaysia-juyvx/giant-clam
- Universiti Teknologi Malaysia. (2023).  Tioman Giant Clams Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/universiti-teknologi-malaysia-juyvx/tioman-giant-clams
- Jacob Solawetz. (2023).  Fish Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/roboflow-gw7yv/fish-yzfml
- Dataset. (2022).  Dataset Dataset  [ Open Source Dataset ]. In  Roboflow Universe .  Roboflow . https://universe.roboflow.com/dataset-gdypo/dataset-axhm3
- Addison Howard, W. K., Eunbyung Park. (2018). ImageNet Object Localization Challenge. Kaggle. https://kaggle.com/competitions/imagenet-object-localization-challenge
- Australian Institute of Marine Science (AIMS), University of Western Australia (UWA) and Curtin University. (2019), OzFish Dataset - Machine learning dataset for Baited Remote Underwater Video Stations, https://doi.org/10.25845/5e28f062c5097
- GBIF.org (09 January 2024) GBIF Occurrence Download  https://doi.org/10.15468/dl.w5xy62
- GBIF.org (09 January 2024) GBIF Occurrence Download  https://doi.org/10.15468/dl.a5uwzp
- GBIF.org (09 January 2024) GBIF Occurrence Download  https://doi.org/10.15468/dl.r5xqkc
- GBIF.org (09 January 2024) GBIF Occurrence Download  https://doi.org/10.15468/dl.ug7n62
- GBIF.org (19 December 2023) GBIF Occurrence Download  https://doi.org/10.15468/dl.32mwtb


### Model

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO (Version 8.0.0) [Computer software]. https://github.com/ultralytics/ultralytics