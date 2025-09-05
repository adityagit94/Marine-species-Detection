# Changelog

All notable changes to the Marine Detect project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Performance monitoring and metrics collection
- Advanced error handling and logging
- Model versioning system
- Batch processing optimization

### Changed
- Improved API response format
- Enhanced documentation structure

### Fixed
- Memory leak in video processing
- Thread safety issues in batch processing

## [1.0.0] - 2024-01-15

### Added
- Initial release of Marine Detect system
- YOLOv8-based marine species detection models
- Fish and Invertebrates detection model (15 classes)
- MegaFauna detection model (3 classes)
- FastAPI-based REST API
- Command-line interface (CLI)
- Docker containerization support
- Comprehensive documentation
- Interactive Jupyter tutorial
- Example scripts and educational content
- Automated setup script for students
- Professional project structure
- Unit and integration test suite
- CI/CD pipeline with GitHub Actions

### Models
- **Fish and Invertebrates Model**:
  - 15 species classes
  - mAP@0.5: 0.867
  - Trained on 12,243 images
  - Optimized confidence threshold: 0.522

- **MegaFauna Model**:
  - 3 species classes (sharks, rays, turtles)
  - mAP@0.5: 0.851
  - Trained on 8,130 images
  - Optimized confidence threshold: 0.6

### API Endpoints
- `GET /health` - Health check
- `GET /models` - Model information
- `POST /detect` - Single image detection
- `POST /detect/batch` - Batch image processing
- `GET /stats` - Performance statistics

### CLI Commands
- `marine-detect detect-images` - Process images
- `marine-detect detect-video` - Process videos
- `marine-detect serve` - Start API server
- `marine-detect setup` - Environment setup

### Documentation
- Complete installation guide
- API documentation with examples
- Development guide with best practices
- Model documentation with performance metrics
- Troubleshooting guide
- Deployment guide for various platforms
- Architecture overview

### Educational Content
- Interactive Jupyter notebook tutorial
- Learning exercises for different skill levels
- Comprehensive learning resources
- Example scripts with detailed comments
- Quick start guide for students

### Performance
- Average inference time: ~2.8ms (GPU), ~45ms (CPU)
- Memory usage: ~1-2GB depending on model and batch size
- Supports real-time processing for video streams
- Optimized for both accuracy and speed

### Supported Formats
- **Images**: JPG, JPEG, PNG, BMP
- **Videos**: MP4, AVI, MOV, MKV
- **Input sizes**: Up to 1920x1080 (configurable)
- **Batch processing**: Multiple images and videos

### Platform Support
- **Operating Systems**: Linux, macOS, Windows
- **Python**: 3.8, 3.9, 3.10, 3.11
- **Hardware**: CPU and GPU (CUDA) support
- **Deployment**: Docker, Kubernetes, Cloud platforms

### Dependencies
- PyTorch >= 1.9.0
- Ultralytics YOLO >= 8.0.200
- OpenCV >= 4.8.0
- FastAPI >= 0.95.0
- NumPy >= 1.24.4
- Pillow >= 10.0.0

## [0.9.0] - 2024-01-01 (Beta)

### Added
- Initial beta release
- Basic detection functionality
- Preliminary model training
- Core API structure

### Known Issues
- Limited species coverage
- Performance optimization needed
- Documentation incomplete

## [0.1.0] - 2023-12-01 (Alpha)

### Added
- Project initialization
- Basic YOLO integration
- Initial model experiments
- Development environment setup

---

## Migration Guides

### Migrating from v0.9.0 to v1.0.0

1. **API Changes**:
   ```python
   # Old format
   response = requests.post('/predict', files={'image': image_file})
   
   # New format
   response = requests.post('/detect', files={'file': image_file})
   ```

2. **Model Loading**:
   ```python
   # Old method
   from marine_detect import MarineDetector
   detector = MarineDetector('model.pt')
   
   # New method
   from marine_detect.predict import predict_on_images
   predict_on_images(model_paths=['model.pt'], ...)
   ```

3. **Configuration**:
   - Environment variables now use `MARINE_DETECT_` prefix
   - Configuration file format changed from JSON to YAML
   - Model paths are now relative to project root

### Breaking Changes

#### v1.0.0
- API endpoint `/predict` renamed to `/detect`
- Response format changed to include additional metadata
- Model file structure reorganized
- Configuration file format updated

#### v0.9.0
- Initial API structure established
- Model format standardized

---

## Development Notes

### Release Process

1. **Version Bumping**:
   ```bash
   # Update version in setup.py, __init__.py, and pyproject.toml
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```

2. **Documentation Updates**:
   - Update CHANGELOG.md
   - Review and update README.md
   - Update API documentation
   - Verify all examples work

3. **Testing**:
   - Run full test suite
   - Test on multiple platforms
   - Verify Docker builds
   - Check deployment scripts

4. **Release**:
   - Create GitHub release
   - Update Docker Hub images
   - Notify users of changes

### Versioning Strategy

- **Major versions** (x.0.0): Breaking changes, major new features
- **Minor versions** (1.x.0): New features, backwards compatible
- **Patch versions** (1.0.x): Bug fixes, small improvements

### Support Policy

- **Current version**: Full support, active development
- **Previous major version**: Security updates only
- **Older versions**: No longer supported

---

## Contributors

### Core Team
- **Aditya Prakash** - Project Lead, Main Developer
  - Email: aditya_2312res46@iitp.ac.in
  - GitHub: [@adityagit94](https://github.com/adityagit94)

### Acknowledgments

- **Data Sources**: Various marine biology datasets and research institutions
- **Model Architecture**: Based on Ultralytics YOLOv8
- **Community**: Contributors, testers, and users who provided feedback

### How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

---

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{prakash2024marine,
  author = {Prakash, Aditya},
  title = {Marine Detect: Object Detection Models for Identifying Species in Marine Environments},
  url = {https://github.com/adityagit94/marine-detect},
  version = {1.0.0},
  year = {2024}
}
```

---

For more information, visit the [project repository](https://github.com/adityagit94/marine-detect) or contact [aditya_2312res46@iitp.ac.in](mailto:aditya_2312res46@iitp.ac.in).
