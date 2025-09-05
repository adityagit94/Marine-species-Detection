# Installation Guide

This guide provides comprehensive installation instructions for the Marine Detect system across different environments and use cases.

## 📋 Prerequisites

### System Requirements

**Minimum Requirements:**
- Python 3.8 or higher
- 4GB RAM
- 2GB free disk space
- CPU with SSE4.2 support

**Recommended Requirements:**
- Python 3.10 or higher
- 8GB+ RAM
- 10GB+ free disk space
- NVIDIA GPU with CUDA support (for training/inference acceleration)
- SSD storage for better I/O performance

### Software Dependencies

- **Python**: 3.8-3.11 (3.10 recommended)
- **Git**: For cloning the repository
- **Docker**: Optional, for containerized deployment
- **CUDA**: Optional, for GPU acceleration

## 🚀 Installation Methods

### Method 1: Quick Install (Recommended for Users)

```bash
# Clone the repository
git clone https://github.com/adityagit94/marine-detect.git
cd marine-detect

# Run the automated setup script
python setup_for_students.py

# Verify installation
python -c "import marine_detect; print('Installation successful!')"
```

### Method 2: Manual Installation

#### Step 1: Clone Repository
```bash
git clone https://github.com/adityagit94/marine-detect.git
cd marine-detect
```

#### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv marine-detect-env
source marine-detect-env/bin/activate  # On Windows: marine-detect-env\Scripts\activate

# Or using conda
conda create -n marine-detect python=3.10
conda activate marine-detect
```

#### Step 3: Install Dependencies
```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

### Method 3: Docker Installation

#### Prerequisites
- Docker Engine 20.10+
- Docker Compose 2.0+

#### Quick Start with Docker
```bash
# Clone repository
git clone https://github.com/adityagit94/marine-detect.git
cd marine-detect

# Build and run with Docker Compose
docker-compose up --build

# Or build manually
docker build -t marine-detect:latest .
docker run -p 8000:8000 marine-detect:latest
```

#### GPU Support (NVIDIA Docker)
```bash
# Build with GPU support
docker build -f Dockerfile.gpu -t marine-detect:gpu .

# Run with GPU
docker run --gpus all -p 8000:8000 marine-detect:gpu
```

### Method 4: Development Installation

For contributors and developers:

```bash
# Clone with development branch
git clone -b develop https://github.com/adityagit94/marine-detect.git
cd marine-detect

# Install with development dependencies
pip install -e ".[dev,api,web]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify installation
pytest tests/
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Model Configuration
FISH_MODEL_PATH=models/fish_model.pt
MEGAFAUNA_MODEL_PATH=models/megafauna_model.pt

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/marine-detect.log

# Performance Settings
BATCH_SIZE=1
MAX_IMAGE_SIZE=1920
DEVICE=cpu  # or cuda for GPU

# Database (optional)
DATABASE_URL=sqlite:///marine_detect.db
```

### Model Setup

1. **Download Pre-trained Models** (if available):
   ```bash
   # Create models directory
   mkdir -p models
   
   # Download models (replace with actual URLs)
   wget -O models/fish_model.pt "MODEL_DOWNLOAD_URL"
   wget -O models/megafauna_model.pt "MODEL_DOWNLOAD_URL"
   ```

2. **Verify Model Files**:
   ```bash
   python -c "
   from ultralytics import YOLO
   model = YOLO('models/fish_model.pt')
   print(f'Model loaded: {len(model.names)} classes')
   "
   ```

## ✅ Verification

### Basic Functionality Test
```bash
# Test CLI
marine-detect --help

# Test API (if installed)
marine-detect serve &
curl http://localhost:8000/health

# Test Python import
python -c "
from marine_detect.predict import predict_on_images
print('Marine Detect successfully installed!')
"
```

### Run Example
```bash
# Run basic detection example
python examples/basic_detection.py

# Run interactive tutorial
jupyter notebook tutorial.ipynb
```

## 🐛 Troubleshooting Installation

### Common Issues

#### Issue: ImportError for OpenCV
```bash
# Solution: Install OpenCV properly
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python==4.8.0.74
```

#### Issue: CUDA not detected
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Issue: Permission denied on Linux/Mac
```bash
# Fix permissions
sudo chown -R $USER:$USER marine-detect/
chmod +x setup_for_students.py
```

#### Issue: Memory errors during installation
```bash
# Use pip with no cache
pip install --no-cache-dir -r requirements.txt

# Or install packages individually
pip install ultralytics
pip install opencv-python
# ... continue with other packages
```

### Platform-Specific Issues

#### Windows
- Install Visual C++ Redistributable if needed
- Use Windows Subsystem for Linux (WSL) for better compatibility
- Ensure Python is added to PATH

#### macOS
- Install Xcode Command Line Tools: `xcode-select --install`
- Use Homebrew for system dependencies: `brew install python@3.10`

#### Linux
- Install system dependencies:
  ```bash
  # Ubuntu/Debian
  sudo apt-get update
  sudo apt-get install python3-dev python3-pip libgl1-mesa-glx

  # CentOS/RHEL
  sudo yum install python3-devel python3-pip mesa-libGL
  ```

## 🔄 Updating

### Update to Latest Version
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Reinstall package
pip install -e . --force-reinstall
```

### Migration Guide
When updating between major versions, check the [CHANGELOG.md](../CHANGELOG.md) for breaking changes and migration instructions.

## 🗑️ Uninstallation

### Remove Package
```bash
# Uninstall package
pip uninstall marine-detect

# Remove virtual environment
rm -rf marine-detect-env/  # or conda env remove -n marine-detect

# Remove cloned repository
rm -rf marine-detect/
```

### Clean Docker Installation
```bash
# Remove containers and images
docker-compose down --rmi all --volumes --remove-orphans
docker system prune -a
```

## 📞 Support

If you encounter installation issues:

1. Check the [Troubleshooting Guide](troubleshooting.md)
2. Search [existing issues](https://github.com/adityagit94/marine-detect/issues)
3. Create a [new issue](https://github.com/adityagit94/marine-detect/issues/new) with:
   - Operating system and version
   - Python version
   - Complete error message
   - Installation method used

---

**Next Steps**: After successful installation, check out the [Quick Start Guide](../README.md#quick-start) or explore the [API Documentation](api.md).
