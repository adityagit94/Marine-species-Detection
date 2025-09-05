# Troubleshooting Guide

This guide helps you diagnose and resolve common issues with the Marine Detect system. Issues are organized by category with step-by-step solutions.

## 🔍 Quick Diagnostics

### System Health Check

Run this diagnostic script to check your system:

```bash
# Check system requirements
python -c "
import sys
import torch
import cv2
import numpy as np
from ultralytics import YOLO

print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'OpenCV: {cv2.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
print('✅ All core dependencies available')
"
```

### Environment Verification

```bash
# Verify installation
marine-detect --help

# Check API health
marine-detect serve &
curl http://localhost:8000/health
```

## 🐛 Installation Issues

### Issue: ModuleNotFoundError

**Symptoms:**
```
ModuleNotFoundError: No module named 'marine_detect'
```

**Solutions:**

1. **Install in development mode:**
   ```bash
   pip install -e .
   ```

2. **Check Python path:**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

3. **Verify virtual environment:**
   ```bash
   which python
   pip list | grep marine-detect
   ```

### Issue: OpenCV Import Error

**Symptoms:**
```
ImportError: libGL.so.1: cannot open shared object file
```

**Solutions:**

1. **Ubuntu/Debian:**
   ```bash
   sudo apt-get update
   sudo apt-get install libgl1-mesa-glx libglib2.0-0
   ```

2. **CentOS/RHEL:**
   ```bash
   sudo yum install mesa-libGL glib2
   ```

3. **Alternative OpenCV installation:**
   ```bash
   pip uninstall opencv-python opencv-contrib-python
   pip install opencv-python-headless
   ```

### Issue: CUDA/GPU Problems

**Symptoms:**
```
RuntimeError: CUDA out of memory
RuntimeError: No CUDA GPUs are available
```

**Solutions:**

1. **Check CUDA installation:**
   ```bash
   nvidia-smi
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Install CUDA-compatible PyTorch:**
   ```bash
   # For CUDA 11.8
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   
   # For CUDA 12.1
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Force CPU usage:**
   ```python
   # In your code
   import torch
   device = 'cpu'  # Force CPU
   model = YOLO('model.pt')
   results = model(image, device=device)
   ```

4. **Reduce batch size:**
   ```python
   # Process images one at a time
   for image in images:
       result = model(image)
   ```

## 🖼️ Image Processing Issues

### Issue: Poor Detection Performance

**Symptoms:**
- No detections found
- Low confidence scores
- Missing obvious species

**Diagnosis:**
```python
def diagnose_image_issues(image_path):
    """Diagnose common image issues."""
    import cv2
    import numpy as np
    
    image = cv2.imread(image_path)
    if image is None:
        return "❌ Cannot load image"
    
    h, w, c = image.shape
    print(f"Image dimensions: {w}x{h}x{c}")
    
    # Check image quality
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    print(f"Blur score: {blur_score:.2f} (>100 is good)")
    
    # Check brightness
    brightness = np.mean(gray)
    print(f"Brightness: {brightness:.2f} (50-200 is good)")
    
    # Check contrast
    contrast = gray.std()
    print(f"Contrast: {contrast:.2f} (>30 is good)")
    
    return "✅ Image diagnosis complete"

# Usage
diagnose_image_issues("your_image.jpg")
```

**Solutions:**

1. **Adjust confidence threshold:**
   ```python
   # Lower threshold for more detections
   results = model(image, conf=0.3)  # Instead of 0.5
   ```

2. **Image preprocessing:**
   ```python
   import cv2
   
   def preprocess_image(image_path):
       image = cv2.imread(image_path)
       
       # Enhance contrast
       lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
       l, a, b = cv2.split(lab)
       clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
       l = clahe.apply(l)
       enhanced = cv2.merge([l, a, b])
       enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
       
       return enhanced
   ```

3. **Check image format:**
   ```python
   # Ensure RGB format
   image = cv2.imread(image_path)
   image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   results = model(image_rgb)
   ```

### Issue: Memory Errors

**Symptoms:**
```
RuntimeError: CUDA out of memory
MemoryError: Unable to allocate array
```

**Solutions:**

1. **Reduce image size:**
   ```python
   def resize_image(image, max_size=1920):
       h, w = image.shape[:2]
       if max(h, w) > max_size:
           scale = max_size / max(h, w)
           new_w = int(w * scale)
           new_h = int(h * scale)
           image = cv2.resize(image, (new_w, new_h))
       return image
   ```

2. **Process images sequentially:**
   ```python
   # Instead of batch processing
   for image_file in image_files:
       image = cv2.imread(image_file)
       results = model(image)
       # Process results
       del image, results  # Free memory
       gc.collect()
   ```

3. **Monitor memory usage:**
   ```python
   import psutil
   
   def check_memory():
       memory = psutil.virtual_memory()
       print(f"Memory usage: {memory.percent}%")
       print(f"Available: {memory.available / (1024**3):.1f} GB")
   ```

## 🌐 API Issues

### Issue: API Server Won't Start

**Symptoms:**
```
Address already in use
Permission denied
```

**Solutions:**

1. **Check port availability:**
   ```bash
   # Check if port 8000 is in use
   lsof -i :8000
   
   # Kill process using the port
   kill -9 <PID>
   
   # Or use different port
   marine-detect serve --port 8001
   ```

2. **Permission issues:**
   ```bash
   # Use port > 1024 for non-root users
   marine-detect serve --port 8080
   ```

3. **Check firewall:**
   ```bash
   # Ubuntu/Debian
   sudo ufw allow 8000
   
   # CentOS/RHEL
   sudo firewall-cmd --add-port=8000/tcp --permanent
   sudo firewall-cmd --reload
   ```

### Issue: API Requests Failing

**Symptoms:**
- Connection refused
- Timeout errors
- 500 Internal Server Error

**Solutions:**

1. **Test with curl:**
   ```bash
   # Test health endpoint
   curl -v http://localhost:8000/health
   
   # Test with image
   curl -X POST "http://localhost:8000/detect" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpg"
   ```

2. **Check API logs:**
   ```bash
   # Run with verbose logging
   LOG_LEVEL=DEBUG marine-detect serve
   ```

3. **Validate request format:**
   ```python
   import requests
   
   def test_api():
       # Health check
       response = requests.get("http://localhost:8000/health")
       print(f"Health: {response.status_code}")
       
       # Image detection
       with open("test_image.jpg", "rb") as f:
           files = {"file": f}
           response = requests.post(
               "http://localhost:8000/detect",
               files=files
           )
       print(f"Detection: {response.status_code}")
       return response.json()
   ```

## 🐳 Docker Issues

### Issue: Docker Build Failures

**Symptoms:**
```
Error building image
Package not found
Permission denied
```

**Solutions:**

1. **Check Docker version:**
   ```bash
   docker --version
   docker-compose --version
   ```

2. **Clean Docker cache:**
   ```bash
   docker system prune -a
   docker builder prune
   ```

3. **Build with no cache:**
   ```bash
   docker build --no-cache -t marine-detect:latest .
   ```

4. **Check Dockerfile syntax:**
   ```dockerfile
   # Ensure proper syntax
   FROM python:3.10-slim
   
   # Set working directory before COPY
   WORKDIR /app
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   ```

### Issue: Container Runtime Problems

**Symptoms:**
- Container exits immediately
- Cannot access API from host
- Volume mount issues

**Solutions:**

1. **Check container logs:**
   ```bash
   docker logs <container_id>
   docker run -it marine-detect:latest bash  # Interactive mode
   ```

2. **Port mapping:**
   ```bash
   # Ensure proper port mapping
   docker run -p 8000:8000 marine-detect:latest
   ```

3. **Volume permissions:**
   ```bash
   # Fix volume permissions
   sudo chown -R $(id -u):$(id -g) ./data
   docker run -v $(pwd)/data:/app/data marine-detect:latest
   ```

## 📊 Performance Issues

### Issue: Slow Inference

**Symptoms:**
- Long processing times
- High CPU/memory usage
- Timeouts

**Solutions:**

1. **Profile performance:**
   ```python
   import time
   import cProfile
   
   def profile_inference():
       profiler = cProfile.Profile()
       profiler.enable()
       
       # Your inference code
       results = model(image)
       
       profiler.disable()
       profiler.print_stats(sort='cumulative')
   ```

2. **Optimize image size:**
   ```python
   # Resize to model input size
   image = cv2.resize(image, (640, 640))
   results = model(image)
   ```

3. **Use GPU acceleration:**
   ```python
   # Ensure GPU usage
   model = YOLO('model.pt')
   model.to('cuda')
   results = model(image, device='cuda')
   ```

4. **Batch processing:**
   ```python
   # Process multiple images together
   results = model([image1, image2, image3])
   ```

### Issue: High Memory Usage

**Solutions:**

1. **Monitor memory:**
   ```python
   import gc
   import torch
   
   def clear_memory():
       gc.collect()
       if torch.cuda.is_available():
           torch.cuda.empty_cache()
   ```

2. **Optimize model loading:**
   ```python
   # Load model once, reuse
   model = YOLO('model.pt')
   
   for image in images:
       results = model(image)
       # Process results immediately
       clear_memory()
   ```

## 🔧 Model-Specific Issues

### Issue: Model Files Not Found

**Symptoms:**
```
FileNotFoundError: model.pt not found
```

**Solutions:**

1. **Check model paths:**
   ```python
   import os
   from pathlib import Path
   
   model_path = "models/fish_model.pt"
   if not Path(model_path).exists():
       print(f"❌ Model not found: {model_path}")
       print(f"Current directory: {os.getcwd()}")
       print(f"Available files: {list(Path('models/').glob('*.pt'))}")
   ```

2. **Download models:**
   ```bash
   # Create models directory
   mkdir -p models
   
   # Download pre-trained models (replace with actual URLs)
   wget -O models/fish_model.pt "MODEL_URL"
   ```

### Issue: Poor Model Performance

**Solutions:**

1. **Check model version:**
   ```python
   from ultralytics import YOLO
   
   model = YOLO('model.pt')
   print(f"Model: {model.model}")
   print(f"Classes: {model.names}")
   ```

2. **Validate input format:**
   ```python
   # Ensure correct input format
   image = cv2.imread(image_path)
   image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   results = model(image_rgb)
   ```

## 📱 Platform-Specific Issues

### Windows Issues

1. **Path separators:**
   ```python
   import os
   model_path = os.path.join("models", "fish_model.pt")
   ```

2. **Long path support:**
   ```bash
   # Enable long paths in Windows
   git config --system core.longpaths true
   ```

### macOS Issues

1. **Permissions:**
   ```bash
   # Fix permissions
   chmod +x setup_for_students.py
   ```

2. **Apple Silicon compatibility:**
   ```bash
   # Install compatible packages
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   ```

### Linux Issues

1. **System dependencies:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-dev libgl1-mesa-glx
   
   # CentOS/RHEL
   sudo yum install python3-devel mesa-libGL
   ```

## 🆘 Getting Help

### Before Asking for Help

1. **Check this troubleshooting guide**
2. **Search existing issues**: [GitHub Issues](https://github.com/adityagit94/marine-detect/issues)
3. **Try the diagnostic commands** provided above
4. **Check the documentation**: [API Docs](api.md), [Installation Guide](installation.md)

### Creating a Bug Report

When reporting issues, include:

```markdown
## Environment
- OS: [e.g., Ubuntu 20.04, Windows 10, macOS 12]
- Python version: [e.g., 3.10.2]
- Marine Detect version: [e.g., 1.0.0]
- Installation method: [pip, Docker, source]

## Issue Description
Brief description of the problem.

## Steps to Reproduce
1. Step one
2. Step two
3. Step three

## Expected Behavior
What you expected to happen.

## Actual Behavior
What actually happened.

## Error Messages
```
Full error traceback here
```

## Additional Context
Any other relevant information.
```

### Contact Information

- **GitHub Issues**: [Create an issue](https://github.com/adityagit94/marine-detect/issues/new)
- **Email**: [aditya_2312res46@iitp.ac.in](mailto:aditya_2312res46@iitp.ac.in)
- **Documentation**: [Project Documentation](README.md)

## 🔄 Emergency Recovery

### Complete Reset

If all else fails, try a complete reset:

```bash
# Remove virtual environment
rm -rf venv/

# Clean pip cache
pip cache purge

# Fresh installation
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# Verify installation
python -c "import marine_detect; print('✅ Installation successful')"
```

### Docker Reset

```bash
# Remove all containers and images
docker system prune -a --volumes

# Rebuild from scratch
docker build --no-cache -t marine-detect:latest .
```

---

**Still having issues?** Don't hesitate to [create an issue](https://github.com/adityagit94/marine-detect/issues/new) or contact [aditya_2312res46@iitp.ac.in](mailto:aditya_2312res46@iitp.ac.in)
