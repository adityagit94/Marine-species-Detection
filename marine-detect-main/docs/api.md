# API Documentation

The Marine Detect API provides REST endpoints for marine species detection using YOLOv8 models. This documentation covers all available endpoints, request/response formats, and usage examples.

## 🚀 Quick Start

### Starting the API Server

```bash
# Using CLI
marine-detect serve --host 0.0.0.0 --port 8000

# Using Python
python -m marine_detect.api

# Using Docker
docker run -p 8000:8000 marine-detect:latest
```

### Base URL
```
http://localhost:8000
```

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📋 API Overview

### Authentication
Currently, the API does not require authentication. For production deployments, implement JWT or API key authentication.

### Content Types
- **Request**: `multipart/form-data` (for file uploads), `application/json`
- **Response**: `application/json`

### Rate Limiting
No rate limiting is currently implemented. Consider adding rate limiting for production use.

## 🛠️ Endpoints

### Health Check

#### `GET /health`
Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T12:00:00Z",
  "version": "1.0.0"
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/health"
```

---

### Root Information

#### `GET /`
Get API information and available endpoints.

**Response:**
```json
{
  "message": "Marine Species Detection API",
  "version": "1.0.0",
  "author": "Aditya Prakash",
  "endpoints": {
    "/detect": "POST - Upload image for species detection",
    "/health": "GET - Health check",
    "/models": "GET - Available models information"
  }
}
```

---

### Model Information

#### `GET /models`
Get information about available models.

**Response:**
```json
{
  "available_models": ["fish_invertebrates", "megafauna"],
  "model_details": {
    "fish_invertebrates": {
      "classes": 15,
      "description": "Detects fish and invertebrate species",
      "supported_species": [
        "fish", "serranidae", "urchin", "scaridae", "chaetodontidae",
        "giant_clam", "lutjanidae", "muraenidae", "sea_cucumber",
        "haemulidae", "lobster", "crown_of_thorns", "bolbometopon_muricatum",
        "cheilinus_undulatus", "cromileptes_altivelis"
      ]
    },
    "megafauna": {
      "classes": 3,
      "description": "Detects large marine species",
      "supported_species": ["ray", "shark", "turtle"]
    }
  }
}
```

---

### Species Detection

#### `POST /detect`
Detect marine species in uploaded image.

**Parameters:**
- `file` (required): Image file (multipart/form-data)
- `confidence_threshold` (optional): Detection confidence threshold (0.0-1.0, default: 0.5)
- `model_type` (optional): Model to use ("fish", "megafauna", "both", default: "both")
- `return_image` (optional): Include annotated image in response (boolean, default: true)
- `save_results` (optional): Save results to file (boolean, default: false)

**Request Example:**
```bash
curl -X POST "http://localhost:8000/detect" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@marine_image.jpg" \
  -F "confidence_threshold=0.6" \
  -F "model_type=both"
```

**Response:**
```json
{
  "filename": "marine_image.jpg",
  "detections": [
    {
      "class_name": "turtle",
      "confidence": 0.85,
      "bbox": [100, 100, 200, 200],
      "class_id": 2,
      "area": 10000,
      "center": [150, 150]
    },
    {
      "class_name": "fish",
      "confidence": 0.72,
      "bbox": [300, 150, 400, 250],
      "class_id": 0,
      "area": 10000,
      "center": [350, 200]
    }
  ],
  "total_detections": 2,
  "confidence_threshold": 0.6,
  "model_type": "both",
  "processing_time": 1.23,
  "image_dimensions": {
    "width": 1920,
    "height": 1080
  },
  "result_image": "base64_encoded_image_string",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

---

### Batch Detection

#### `POST /detect/batch`
Process multiple images at once.

**Parameters:**
- `files` (required): Multiple image files
- `confidence_threshold` (optional): Detection confidence threshold (default: 0.5)
- `model_type` (optional): Model to use (default: "both")

**Request Example:**
```bash
curl -X POST "http://localhost:8000/detect/batch" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "confidence_threshold=0.5"
```

**Response:**
```json
{
  "batch_id": "batch_123456",
  "total_images": 2,
  "results": [
    {
      "filename": "image1.jpg",
      "detections": [...],
      "total_detections": 3
    },
    {
      "filename": "image2.jpg",
      "detections": [...],
      "total_detections": 1
    }
  ],
  "summary": {
    "total_detections": 4,
    "average_confidence": 0.78,
    "processing_time": 2.45
  }
}
```

---

### Model Statistics

#### `GET /stats`
Get model performance statistics.

**Response:**
```json
{
  "fish_model": {
    "total_inferences": 1250,
    "average_processing_time": 0.85,
    "accuracy_metrics": {
      "mAP50": 0.823,
      "mAP50-95": 0.745
    },
    "class_distribution": {
      "fish": 450,
      "turtle": 320,
      "shark": 180
    }
  },
  "system_stats": {
    "uptime": "2 days, 14:30:22",
    "cpu_usage": 45.2,
    "memory_usage": 68.5,
    "gpu_usage": 23.1
  }
}
```

## 🔧 Advanced Usage

### Python Client Example

```python
import requests
import json
from pathlib import Path

class MarineDetectClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def detect_image(self, image_path, confidence=0.5, model_type="both"):
        """Detect species in a single image."""
        url = f"{self.base_url}/detect"
        
        with open(image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'confidence_threshold': confidence,
                'model_type': model_type
            }
            response = requests.post(url, files=files, data=data)
        
        return response.json()
    
    def health_check(self):
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Usage
client = MarineDetectClient()
result = client.detect_image("marine_image.jpg", confidence=0.6)
print(f"Found {result['total_detections']} marine species")
```

### JavaScript Client Example

```javascript
class MarineDetectClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async detectImage(imageFile, confidence = 0.5, modelType = 'both') {
        const formData = new FormData();
        formData.append('file', imageFile);
        formData.append('confidence_threshold', confidence);
        formData.append('model_type', modelType);
        
        const response = await fetch(`${this.baseUrl}/detect`, {
            method: 'POST',
            body: formData
        });
        
        return await response.json();
    }
    
    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        return await response.json();
    }
}

// Usage
const client = new MarineDetectClient();
const fileInput = document.getElementById('imageInput');
const result = await client.detectImage(fileInput.files[0], 0.6);
console.log(`Found ${result.total_detections} marine species`);
```

## 📊 Response Schemas

### Detection Object
```json
{
  "class_name": "string",
  "confidence": "float (0.0-1.0)",
  "bbox": "array[float] [x1, y1, x2, y2]",
  "class_id": "integer",
  "area": "float",
  "center": "array[float] [x, y]"
}
```

### Error Response
```json
{
  "error": {
    "code": "string",
    "message": "string",
    "details": "object"
  },
  "timestamp": "string (ISO 8601)"
}
```

## ❌ Error Handling

### HTTP Status Codes

- `200`: Success
- `400`: Bad Request (invalid parameters)
- `413`: Payload Too Large (file size limit exceeded)
- `415`: Unsupported Media Type (invalid file format)
- `422`: Unprocessable Entity (validation error)
- `500`: Internal Server Error

### Error Examples

#### Invalid File Format
```json
{
  "error": {
    "code": "INVALID_FILE_FORMAT",
    "message": "Unsupported file format. Supported formats: jpg, jpeg, png, bmp",
    "details": {
      "received_format": "gif",
      "supported_formats": ["jpg", "jpeg", "png", "bmp"]
    }
  }
}
```

#### File Too Large
```json
{
  "error": {
    "code": "FILE_TOO_LARGE",
    "message": "File size exceeds maximum limit of 10MB",
    "details": {
      "file_size": 15728640,
      "max_size": 10485760
    }
  }
}
```

## 🔒 Security Considerations

### Input Validation
- File size limits (default: 10MB)
- File type validation
- Image dimension limits
- Parameter validation

### Recommended Security Headers
```python
# Add to your reverse proxy or API gateway
headers = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
}
```

## 🚀 Performance Tips

1. **Batch Processing**: Use `/detect/batch` for multiple images
2. **Confidence Tuning**: Adjust thresholds based on your use case
3. **Model Selection**: Use specific models ("fish" or "megafauna") when possible
4. **Image Optimization**: Resize images to optimal dimensions before upload
5. **Caching**: Implement response caching for repeated requests

## 📞 Support

For API-related issues:
- Check the interactive documentation at `/docs`
- Review error messages and status codes
- Contact: [aditya_2312res46@iitp.ac.in](mailto:aditya_2312res46@iitp.ac.in)

---

**Next**: [Development Guide](development.md) | [Model Documentation](models.md)
