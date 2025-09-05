# Development Guide

This guide provides comprehensive information for developers who want to contribute to the Marine Detect project or use it as a foundation for their own work.

## 🎯 Development Philosophy

### Core Principles
- **Code Quality**: Write clean, readable, and maintainable code
- **Testing**: Comprehensive test coverage with unit and integration tests
- **Documentation**: Clear documentation for all public APIs and complex logic
- **Performance**: Optimize for both accuracy and inference speed
- **Reproducibility**: Ensure consistent results across different environments

### Technology Stack
- **Language**: Python 3.8+
- **ML Framework**: PyTorch, Ultralytics YOLO
- **Web Framework**: FastAPI
- **Testing**: pytest, pytest-cov
- **Code Quality**: black, flake8, mypy, pre-commit
- **Documentation**: Sphinx, MkDocs
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions

## 🏗️ Architecture Overview

### Project Structure
```
marine-detect/
├── src/marine_detect/          # Main package
│   ├── __init__.py
│   ├── predict.py              # Core inference logic
│   ├── api.py                  # FastAPI application
│   ├── cli.py                  # Command-line interface
│   ├── models.py               # Data models and schemas
│   ├── config.py               # Configuration management
│   └── utils/                  # Utility modules
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   ├── integration/            # Integration tests
│   └── fixtures/               # Test data and fixtures
├── docs/                       # Documentation
├── examples/                   # Example scripts
├── assets/                     # Media assets
├── models/                     # Model files (not in git)
├── data/                       # Data files (not in git)
└── scripts/                    # Utility scripts
```

### Core Components

#### 1. Prediction Engine (`predict.py`)
- **Purpose**: Core inference logic for marine species detection
- **Key Functions**:
  - `predict_on_images()`: Batch image processing
  - `predict_on_video()`: Video processing
  - `save_combined_image()`: Result visualization

#### 2. API Layer (`api.py`)
- **Purpose**: REST API for model serving
- **Framework**: FastAPI with automatic OpenAPI documentation
- **Features**: File upload, batch processing, error handling

#### 3. CLI Interface (`cli.py`)
- **Purpose**: Command-line tool for various operations
- **Framework**: Click for argument parsing
- **Commands**: detect, serve, setup, train

#### 4. Data Models (`models.py`)
- **Purpose**: Pydantic models for data validation
- **Models**: Detection, BoundingBox, ModelConfig

## 🛠️ Development Setup

### Prerequisites
- Python 3.8+ (3.10 recommended)
- Git
- Docker (optional)
- CUDA toolkit (for GPU development)

### Initial Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/marine-detect.git
   cd marine-detect
   ```

2. **Create Development Environment**
   ```bash
   # Using conda (recommended)
   conda create -n marine-detect-dev python=3.10
   conda activate marine-detect-dev
   
   # Or using venv
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   # Install in development mode with all extras
   pip install -e ".[dev,api,web]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

4. **Verify Installation**
   ```bash
   # Run tests
   pytest
   
   # Check code style
   black --check src tests
   flake8 src tests
   mypy src
   ```

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Write code following the style guide
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run full test suite
   pytest tests/ -v
   
   # Run specific test file
   pytest tests/test_predict.py -v
   
   # Run with coverage
   pytest --cov=src --cov-report=html
   ```

4. **Code Quality Checks**
   ```bash
   # Format code
   black src tests
   
   # Check linting
   flake8 src tests
   
   # Type checking
   mypy src
   
   # Run all pre-commit hooks
   pre-commit run --all-files
   ```

5. **Commit and Push**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Use the PR template
   - Ensure CI passes
   - Request review from maintainers

## 🧪 Testing Guidelines

### Test Structure
```
tests/
├── conftest.py                 # Pytest configuration and fixtures
├── unit/                       # Unit tests
│   ├── test_predict.py
│   ├── test_api.py
│   └── test_models.py
├── integration/                # Integration tests
│   ├── test_api_integration.py
│   └── test_pipeline.py
└── fixtures/                   # Test data
    ├── images/
    └── models/
```

### Writing Tests

#### Unit Test Example
```python
import pytest
import numpy as np
from unittest.mock import Mock, patch
from marine_detect.predict import save_combined_image

class TestPrediction:
    def test_save_combined_image(self, tmp_path):
        """Test image saving functionality."""
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        # Mock results
        mock_result = Mock()
        mock_result.plot.return_value = test_image
        
        # Test function
        save_combined_image(
            images_input_folder_path=str(tmp_path),
            image_name="test.jpg",
            output_folder_pred_images=str(tmp_path / "output"),
            combined_results=[mock_result]
        )
        
        # Assert output file exists
        assert (tmp_path / "output" / "test.jpg").exists()
```

#### Integration Test Example
```python
import pytest
from fastapi.testclient import TestClient
from marine_detect.api import app

class TestAPIIntegration:
    @pytest.fixture
    def client(self):
        return TestClient(app)
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_detect_endpoint(self, client, sample_image):
        """Test detection endpoint with sample image."""
        files = {"file": ("test.jpg", sample_image, "image/jpeg")}
        response = client.post("/detect", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "detections" in data
        assert "total_detections" in data
```

### Test Commands
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test class
pytest tests/unit/test_predict.py::TestPrediction

# Run tests matching pattern
pytest -k "test_api"

# Run tests with verbose output
pytest -v

# Run tests and stop on first failure
pytest -x

# Run tests in parallel
pytest -n auto
```

## 📝 Code Style Guidelines

### Python Style Guide
We follow PEP 8 with some modifications:

```python
# Line length: 88 characters (Black default)
# Imports: Use isort for consistent import ordering
# Quotes: Double quotes preferred
# Type hints: Required for public APIs

# Example function
def predict_species(
    image: np.ndarray,
    model_path: str,
    confidence_threshold: float = 0.5,
) -> List[Detection]:
    """
    Predict marine species in an image.
    
    Args:
        image: Input image as numpy array
        model_path: Path to the trained model
        confidence_threshold: Minimum confidence for detections
        
    Returns:
        List of detection results
        
    Raises:
        ModelNotFoundError: If model file doesn't exist
        InvalidImageError: If image format is invalid
    """
    # Implementation here
    pass
```

### Documentation Style
- Use Google-style docstrings
- Include type hints for all parameters and return values
- Provide examples for complex functions
- Document exceptions that can be raised

### Commit Message Format
```
type(scope): description

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:
```
feat(api): add batch processing endpoint
fix(predict): handle corrupted image files
docs(readme): update installation instructions
test(api): add integration tests for detection endpoint
```

## 🔧 Configuration Management

### Environment Variables
```python
# config.py
import os
from pathlib import Path
from typing import Optional

class Settings:
    # Model paths
    FISH_MODEL_PATH: str = os.getenv("FISH_MODEL_PATH", "models/fish_model.pt")
    MEGAFAUNA_MODEL_PATH: str = os.getenv("MEGAFAUNA_MODEL_PATH", "models/megafauna_model.pt")
    
    # API settings
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    
    # Performance settings
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "1"))
    MAX_IMAGE_SIZE: int = int(os.getenv("MAX_IMAGE_SIZE", "1920"))
    DEVICE: str = os.getenv("DEVICE", "cpu")
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: Optional[str] = os.getenv("LOG_FILE")

settings = Settings()
```

### Configuration Files
Support for YAML configuration files:

```yaml
# config.yaml
models:
  fish_model_path: "models/fish_model.pt"
  megafauna_model_path: "models/megafauna_model.pt"
  confidence_thresholds:
    fish: 0.5
    megafauna: 0.6

api:
  host: "0.0.0.0"
  port: 8000
  workers: 1

performance:
  batch_size: 4
  max_image_size: 1920
  device: "cuda"

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

## 🚀 Performance Optimization

### Profiling
```python
import cProfile
import pstats
from marine_detect.predict import predict_on_images

# Profile function
profiler = cProfile.Profile()
profiler.enable()

# Your code here
predict_on_images(...)

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)
```

### Memory Optimization
```python
import psutil
import gc

def monitor_memory():
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024  # MB

# Before processing
memory_before = monitor_memory()

# Your processing code
result = process_image(image)

# Clean up
del image
gc.collect()

memory_after = monitor_memory()
print(f"Memory used: {memory_after - memory_before:.2f} MB")
```

## 🐳 Docker Development

### Development Dockerfile
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Install package in development mode
RUN pip install -e ".[dev]"

# Set up git hooks
RUN git config --global --add safe.directory /app
RUN pre-commit install || true

CMD ["bash"]
```

### Docker Compose for Development
```yaml
version: '3.8'

services:
  marine-detect-dev:
    build:
      context: .
      dockerfile: Dockerfile.dev
    volumes:
      - .:/app
      - /app/.git
    ports:
      - "8000:8000"
      - "8888:8888"  # Jupyter
    environment:
      - PYTHONPATH=/app/src
      - LOG_LEVEL=DEBUG
    command: bash
```

## 🔄 Continuous Integration

### GitHub Actions Workflow
```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', 3.11]

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e ".[dev]"
    
    - name: Lint and format check
      run: |
        black --check src tests
        flake8 src tests
        mypy src
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
```

## 📊 Monitoring and Logging

### Logging Setup
```python
import logging
import sys
from pathlib import Path

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Set up logging configuration."""
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # File handler (optional)
    handlers = [console_handler]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers
    )
```

### Performance Monitoring
```python
import time
import functools
from typing import Callable

def monitor_performance(func: Callable) -> Callable:
    """Decorator to monitor function performance."""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            status = "success"
        except Exception as e:
            result = None
            status = "error"
            raise
        finally:
            end_time = time.time()
            logger.info(
                f"{func.__name__} completed in {end_time - start_time:.3f}s "
                f"with status: {status}"
            )
        return result
    
    return wrapper
```

## 🤝 Contributing Guidelines

### Pull Request Process

1. **Fork the repository** and create your feature branch
2. **Write tests** for your changes
3. **Ensure all tests pass** and code coverage is maintained
4. **Update documentation** as needed
5. **Follow code style guidelines**
6. **Write clear commit messages**
7. **Create a pull request** with a clear description

### Pull Request Template
```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Coverage maintained or improved

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

### Code Review Guidelines

**For Authors:**
- Keep PRs focused and small
- Provide clear descriptions
- Respond to feedback promptly
- Test thoroughly before submitting

**For Reviewers:**
- Review code for correctness and style
- Check test coverage
- Verify documentation updates
- Be constructive in feedback

## 📞 Support

For development-related questions:
- Check existing [GitHub issues](https://github.com/adityagit94/marine-detect/issues)
- Join discussions in [GitHub Discussions](https://github.com/adityagit94/marine-detect/discussions)
- Contact: [aditya_2312res46@iitp.ac.in](mailto:aditya_2312res46@iitp.ac.in)

---

**Next**: [Architecture Overview](architecture.md) | [Model Documentation](models.md)
