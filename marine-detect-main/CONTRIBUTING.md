# Contributing to Marine Detect

Thank you for your interest in contributing to the Marine Detect project! This document provides guidelines and information for contributors.

## 🎯 Ways to Contribute

### 1. Code Contributions
- Bug fixes
- New features
- Performance improvements
- Code optimization
- Documentation improvements

### 2. Data Contributions
- Marine species images with annotations
- New species classes
- Dataset validation and cleaning
- Annotation quality improvements

### 3. Documentation
- Tutorial improvements
- API documentation
- Code examples
- Translation to other languages

### 4. Testing
- Bug reports
- Test case improvements
- Performance testing
- Platform compatibility testing

### 5. Community
- Answering questions in issues
- Helping other users
- Creating educational content
- Sharing use cases

## 🚀 Getting Started

### Prerequisites
- Python 3.8+ (3.10 recommended)
- Git
- Basic knowledge of machine learning and computer vision
- Familiarity with PyTorch and YOLO (helpful but not required)

### Development Setup

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/your-username/marine-detect.git
   cd marine-detect
   ```

2. **Set Up Development Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install in development mode
   pip install -e ".[dev,api,web]"
   
   # Install pre-commit hooks
   pre-commit install
   ```

3. **Verify Setup**
   ```bash
   # Run tests
   pytest tests/
   
   # Check code style
   black --check src tests
   flake8 src tests
   mypy src
   ```

4. **Create Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 📝 Development Guidelines

### Code Style

We follow PEP 8 with some modifications enforced by our tools:

- **Line length**: 88 characters (Black default)
- **Imports**: Sorted with isort
- **Type hints**: Required for all public functions
- **Docstrings**: Google style for all public functions and classes

```python
def detect_species(
    image: np.ndarray,
    model_path: str,
    confidence_threshold: float = 0.5,
) -> List[Detection]:
    """
    Detect marine species in an image.
    
    Args:
        image: Input image as numpy array (H, W, C)
        model_path: Path to the trained YOLO model file
        confidence_threshold: Minimum confidence for detections (0.0-1.0)
        
    Returns:
        List of detection results with bounding boxes and confidence scores
        
    Raises:
        FileNotFoundError: If model file doesn't exist
        ValueError: If image format is invalid
        
    Example:
        >>> image = cv2.imread("marine_image.jpg")
        >>> detections = detect_species(image, "models/fish_model.pt", 0.6)
        >>> print(f"Found {len(detections)} marine species")
    """
    # Implementation here
    pass
```

### Commit Message Format

Use conventional commits format:

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks
- `perf`: Performance improvements

**Examples:**
```
feat(api): add batch processing endpoint
fix(predict): handle corrupted image files gracefully
docs(readme): update installation instructions
test(api): add integration tests for detection endpoint
refactor(models): extract common model loading logic
```

### Testing Guidelines

#### Test Structure
```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── unit/                    # Unit tests
│   ├── test_predict.py
│   ├── test_api.py
│   └── test_models.py
├── integration/             # Integration tests
│   ├── test_api_integration.py
│   └── test_pipeline.py
└── fixtures/                # Test data
    ├── images/
    └── models/
```

#### Writing Tests

1. **Unit Tests**: Test individual functions and classes
   ```python
   import pytest
   from marine_detect.predict import save_combined_image
   
   def test_save_combined_image(tmp_path, mock_results):
       """Test image saving functionality."""
       output_path = tmp_path / "output.jpg"
       
       save_combined_image(
           images_input_folder_path=str(tmp_path),
           image_name="test.jpg",
           output_folder_pred_images=str(tmp_path),
           combined_results=mock_results
       )
       
       assert output_path.exists()
   ```

2. **Integration Tests**: Test complete workflows
   ```python
   def test_complete_detection_pipeline(sample_images):
       """Test complete detection pipeline."""
       results = predict_on_images(
           model_paths=["tests/fixtures/models/test_model.pt"],
           confs_threshold=[0.5],
           images_input_folder_path="tests/fixtures/images",
           images_output_folder_path="/tmp/test_output"
       )
       
       assert len(results) > 0
   ```

3. **API Tests**: Test REST endpoints
   ```python
   from fastapi.testclient import TestClient
   from marine_detect.api import app
   
   def test_detect_endpoint():
       """Test detection API endpoint."""
       client = TestClient(app)
       
       with open("tests/fixtures/images/test.jpg", "rb") as f:
           response = client.post(
               "/detect",
               files={"file": ("test.jpg", f, "image/jpeg")}
           )
       
       assert response.status_code == 200
       data = response.json()
       assert "detections" in data
   ```

#### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_predict.py -v

# Run tests matching pattern
pytest -k "test_api" -v

# Run tests and stop on first failure
pytest -x
```

### Documentation Guidelines

#### Code Documentation

1. **Docstrings**: All public functions, classes, and modules must have docstrings
2. **Type Hints**: Use type hints for all function parameters and return values
3. **Comments**: Explain complex logic and algorithms
4. **Examples**: Include usage examples in docstrings

#### External Documentation

1. **README Updates**: Update README.md for user-facing changes
2. **API Documentation**: Update docs/api.md for API changes
3. **Tutorials**: Update or create tutorials for new features
4. **Changelog**: Update CHANGELOG.md for all changes

## 🔄 Pull Request Process

### Before Submitting

1. **Test Your Changes**
   ```bash
   # Run full test suite
   pytest tests/ -v
   
   # Check code style
   black src tests
   flake8 src tests
   mypy src
   
   # Run pre-commit hooks
   pre-commit run --all-files
   ```

2. **Update Documentation**
   - Update relevant documentation files
   - Add or update docstrings
   - Update CHANGELOG.md

3. **Create Quality Commit Messages**
   - Use conventional commit format
   - Include clear descriptions
   - Reference issue numbers if applicable

### Submitting Pull Request

1. **Push Your Changes**
   ```bash
   git add .
   git commit -m "feat: add new marine species detection feature"
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request**
   - Use the PR template
   - Provide clear description of changes
   - Link related issues
   - Add screenshots for UI changes
   - Request review from maintainers

3. **PR Template**
   ```markdown
   ## Description
   Brief description of the changes made.
   
   ## Type of Change
   - [ ] Bug fix (non-breaking change which fixes an issue)
   - [ ] New feature (non-breaking change which adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
   - [ ] Documentation update
   
   ## How Has This Been Tested?
   - [ ] Unit tests
   - [ ] Integration tests
   - [ ] Manual testing
   
   ## Screenshots (if applicable)
   
   ## Checklist
   - [ ] My code follows the style guidelines of this project
   - [ ] I have performed a self-review of my own code
   - [ ] I have commented my code, particularly in hard-to-understand areas
   - [ ] I have made corresponding changes to the documentation
   - [ ] My changes generate no new warnings
   - [ ] I have added tests that prove my fix is effective or that my feature works
   - [ ] New and existing unit tests pass locally with my changes
   - [ ] Any dependent changes have been merged and published in downstream modules
   ```

### Review Process

1. **Automated Checks**: CI/CD pipeline runs automatically
2. **Code Review**: Maintainers review code quality and design
3. **Testing**: Verify that tests pass and functionality works
4. **Documentation**: Check that documentation is updated
5. **Approval**: At least one maintainer approval required
6. **Merge**: Squash and merge into main branch

## 🐛 Reporting Issues

### Bug Reports

Use the bug report template:

```markdown
**Bug Description**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. Ubuntu 20.04, Windows 10, macOS 12]
 - Python Version: [e.g. 3.10.2]
 - Marine Detect Version: [e.g. 1.0.0]
 - Installation Method: [pip, Docker, source]

**Additional Context**
Add any other context about the problem here.
```

### Feature Requests

Use the feature request template:

```markdown
**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is.

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
```

## 📊 Data Contributions

### Image Data Guidelines

1. **Quality Requirements**:
   - Minimum resolution: 640x640 pixels
   - Clear, well-lit images
   - Species clearly visible
   - Minimal motion blur

2. **Annotation Format**:
   - YOLO format (.txt files)
   - Normalized coordinates (0-1)
   - One annotation file per image
   - Consistent class naming

3. **Data Organization**:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   ├── labels/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── dataset.yaml
   ```

4. **Metadata Requirements**:
   - Location information (if available)
   - Date and time
   - Camera/equipment details
   - Environmental conditions

### Dataset Validation

Before submitting datasets:

```python
def validate_dataset(dataset_path):
    """Validate dataset format and quality."""
    # Check file structure
    # Validate annotations
    # Check image quality
    # Verify class consistency
    pass
```

## 🏆 Recognition

### Contributors

All contributors are recognized in:
- README.md contributors section
- CHANGELOG.md acknowledgments
- GitHub contributors page
- Annual contributor highlights

### Contribution Types

We recognize various types of contributions:
- 🐛 Bug reports and fixes
- ✨ New features
- 📖 Documentation improvements
- 🎨 UI/UX improvements
- 🔧 Infrastructure and tooling
- 📊 Data contributions
- 🧪 Testing and QA
- 🌍 Translations
- 💬 Community support

## 📞 Getting Help

### Communication Channels

- **GitHub Issues**: Bug reports, feature requests
- **GitHub Discussions**: General questions, ideas
- **Email**: [aditya_2312res46@iitp.ac.in](mailto:aditya_2312res46@iitp.ac.in)

### Resources

- **Documentation**: [docs/](docs/)
- **Examples**: [examples/](examples/)
- **Tutorials**: [tutorial.ipynb](tutorial.ipynb)
- **API Reference**: [docs/api.md](docs/api.md)

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive environment for all contributors, regardless of:
- Experience level
- Gender identity and expression
- Sexual orientation
- Disability
- Personal appearance
- Body size
- Race
- Ethnicity
- Age
- Religion
- Nationality

### Our Standards

**Positive behaviors include:**
- Using welcoming and inclusive language
- Being respectful of differing viewpoints and experiences
- Gracefully accepting constructive criticism
- Focusing on what is best for the community
- Showing empathy towards other community members

**Unacceptable behaviors include:**
- The use of sexualized language or imagery
- Personal attacks or insulting/derogatory comments
- Trolling or inflammatory comments
- Public or private harassment
- Publishing others' private information without permission
- Other conduct which could reasonably be considered inappropriate

### Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be reported by contacting the project maintainer at [aditya_2312res46@iitp.ac.in](mailto:aditya_2312res46@iitp.ac.in).

## 📄 License

By contributing to Marine Detect, you agree that your contributions will be licensed under the GNU Affero General Public License v3.0.

---

## Thank You!

Your contributions help make Marine Detect better for everyone. Whether you're fixing a typo, adding a feature, or helping other users, every contribution matters.

**Happy coding!** 🌊🐟

---

**Contact**: [aditya_2312res46@iitp.ac.in](mailto:aditya_2312res46@iitp.ac.in)  
**Project**: [https://github.com/adityagit94/marine-detect](https://github.com/adityagit94/marine-detect)
