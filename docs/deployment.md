# Deployment Guide

This comprehensive guide covers deploying the Marine Detect system in various environments, from development to production-scale deployments.

## 🎯 Deployment Overview

### Deployment Options

1. **Local Development**: Single machine setup for development and testing
2. **Docker Container**: Containerized deployment for consistency and portability
3. **Cloud Deployment**: Scalable cloud-based deployment (AWS, GCP, Azure)
4. **Edge Deployment**: Lightweight deployment for edge devices and IoT
5. **Kubernetes**: Container orchestration for large-scale deployments

### Architecture Patterns

- **Single Instance**: Simple deployment for small-scale usage
- **Load Balanced**: Multiple instances behind a load balancer
- **Microservices**: Separate services for different components
- **Serverless**: Function-as-a-Service deployment

## 🐳 Docker Deployment

### Basic Docker Deployment

#### 1. Build Docker Image

```dockerfile
# Dockerfile
FROM python:3.10-slim

LABEL maintainer="Aditya Prakash <aditya_2312res46@iitp.ac.in>"
LABEL description="Marine Species Detection API"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY setup.py README.md ./
COPY models/ ./models/

# Install the package
RUN pip install -e .

# Create non-root user
RUN groupadd -r marine && useradd -r -g marine -d /app -s /sbin/nologin marine && \
    chown -R marine:marine /app
USER marine

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "marine_detect.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. Build and Run

```bash
# Build image
docker build -t marine-detect:latest .

# Run container
docker run -d \
  --name marine-detect-api \
  -p 8000:8000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  --restart unless-stopped \
  marine-detect:latest

# Check logs
docker logs marine-detect-api

# Test deployment
curl http://localhost:8000/health
```

### Docker Compose Deployment

#### docker-compose.yml

```yaml
version: '3.8'

services:
  marine-detect-api:
    build:
      context: .
      dockerfile: Dockerfile
    image: marine-detect:latest
    container_name: marine-detect-api
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models:ro
      - ./data:/app/data
      - ./logs:/app/logs
    environment:
      - LOG_LEVEL=INFO
      - API_WORKERS=1
      - MAX_IMAGE_SIZE=1920
      - DEVICE=cpu
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    container_name: marine-detect-nginx
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - marine-detect-api
    restart: unless-stopped

  redis:
    image: redis:alpine
    container_name: marine-detect-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:

networks:
  default:
    name: marine-detect-network
```

#### nginx.conf

```nginx
events {
    worker_connections 1024;
}

http {
    upstream marine_detect {
        server marine-detect-api:8000;
    }

    server {
        listen 80;
        server_name localhost;

        # Security headers
        add_header X-Frame-Options DENY;
        add_header X-Content-Type-Options nosniff;
        add_header X-XSS-Protection "1; mode=block";

        # File upload size limit
        client_max_body_size 10M;

        location / {
            proxy_pass http://marine_detect;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # Timeout settings
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
        }

        # Static files (if any)
        location /static/ {
            alias /app/static/;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }
    }
}
```

#### Deploy with Docker Compose

```bash
# Deploy all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f marine-detect-api

# Scale API service
docker-compose up -d --scale marine-detect-api=3

# Update deployment
docker-compose pull
docker-compose up -d
```

## ☁️ Cloud Deployment

### AWS Deployment

#### 1. EC2 Instance Setup

```bash
# Launch EC2 instance (Ubuntu 20.04 LTS)
# Instance type: t3.medium or larger
# Security group: Allow ports 22, 80, 443, 8000

# Connect to instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Clone repository
git clone https://github.com/adityagit94/marine-detect.git
cd marine-detect

# Deploy
docker-compose up -d
```

#### 2. ECS Deployment

```json
{
  "family": "marine-detect-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "marine-detect-api",
      "image": "your-account.dkr.ecr.region.amazonaws.com/marine-detect:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "LOG_LEVEL",
          "value": "INFO"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/marine-detect",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": [
          "CMD-SHELL",
          "curl -f http://localhost:8000/health || exit 1"
        ],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

#### 3. Lambda Deployment (Serverless)

```python
# lambda_handler.py
import json
import base64
from io import BytesIO
from PIL import Image
import boto3
from marine_detect.predict import detect_marine_species

def lambda_handler(event, context):
    """AWS Lambda handler for marine species detection."""
    
    try:
        # Parse request
        body = json.loads(event['body'])
        image_data = base64.b64decode(body['image'])
        confidence = body.get('confidence', 0.5)
        
        # Process image
        image = Image.open(BytesIO(image_data))
        detections = detect_marine_species(image, confidence)
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'detections': detections,
                'total_detections': len(detections)
            })
        }
        
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
```

### Google Cloud Platform (GCP)

#### 1. Cloud Run Deployment

```yaml
# cloudbuild.yaml
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/marine-detect:latest', '.']
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/marine-detect:latest']
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
      - 'run'
      - 'deploy'
      - 'marine-detect-api'
      - '--image=gcr.io/$PROJECT_ID/marine-detect:latest'
      - '--platform=managed'
      - '--region=us-central1'
      - '--allow-unauthenticated'
      - '--memory=2Gi'
      - '--cpu=1'
      - '--max-instances=10'
```

```bash
# Deploy to Cloud Run
gcloud builds submit --config cloudbuild.yaml

# Update service
gcloud run deploy marine-detect-api \
  --image gcr.io/PROJECT_ID/marine-detect:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### 2. GKE Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: marine-detect-api
  labels:
    app: marine-detect-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: marine-detect-api
  template:
    metadata:
      labels:
        app: marine-detect-api
    spec:
      containers:
      - name: marine-detect-api
        image: gcr.io/PROJECT_ID/marine-detect:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: marine-detect-service
spec:
  selector:
    app: marine-detect-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Microsoft Azure

#### 1. Container Instances

```bash
# Create resource group
az group create --name marine-detect-rg --location eastus

# Create container instance
az container create \
  --resource-group marine-detect-rg \
  --name marine-detect-api \
  --image your-registry.azurecr.io/marine-detect:latest \
  --cpu 1 \
  --memory 2 \
  --ports 8000 \
  --dns-name-label marine-detect-api \
  --environment-variables LOG_LEVEL=INFO
```

#### 2. App Service

```yaml
# azure-pipelines.yml
trigger:
- main

pool:
  vmImage: 'ubuntu-latest'

variables:
  dockerRegistryServiceConnection: 'your-registry-connection'
  imageRepository: 'marine-detect'
  containerRegistry: 'your-registry.azurecr.io'
  dockerfilePath: '$(Build.SourcesDirectory)/Dockerfile'
  tag: '$(Build.BuildId)'

stages:
- stage: Build
  displayName: Build and push stage
  jobs:
  - job: Build
    displayName: Build
    steps:
    - task: Docker@2
      displayName: Build and push image
      inputs:
        command: buildAndPush
        repository: $(imageRepository)
        dockerfile: $(dockerfilePath)
        containerRegistry: $(dockerRegistryServiceConnection)
        tags: |
          $(tag)
          latest

- stage: Deploy
  displayName: Deploy stage
  dependsOn: Build
  jobs:
  - deployment: Deploy
    displayName: Deploy
    environment: 'production'
    strategy:
      runOnce:
        deploy:
          steps:
          - task: AzureWebAppContainer@1
            displayName: 'Deploy to Azure Web App'
            inputs:
              azureSubscription: 'your-subscription'
              appName: 'marine-detect-api'
              containers: '$(containerRegistry)/$(imageRepository):$(tag)'
```

## 🚀 Kubernetes Deployment

### Complete Kubernetes Manifests

#### 1. Namespace and ConfigMap

```yaml
# kubernetes/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: marine-detect

---
# kubernetes/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: marine-detect-config
  namespace: marine-detect
data:
  LOG_LEVEL: "INFO"
  API_WORKERS: "1"
  MAX_IMAGE_SIZE: "1920"
  DEVICE: "cpu"
```

#### 2. Secret Management

```yaml
# kubernetes/secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: marine-detect-secrets
  namespace: marine-detect
type: Opaque
data:
  # Base64 encoded values
  database-url: <base64-encoded-url>
  api-key: <base64-encoded-key>
```

#### 3. Deployment with PVC

```yaml
# kubernetes/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: marine-detect-api
  namespace: marine-detect
  labels:
    app: marine-detect-api
    version: v1.0.0
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app: marine-detect-api
  template:
    metadata:
      labels:
        app: marine-detect-api
        version: v1.0.0
    spec:
      containers:
      - name: marine-detect-api
        image: marine-detect:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
          name: http
        env:
        - name: LOG_LEVEL
          valueFrom:
            configMapKeyRef:
              name: marine-detect-config
              key: LOG_LEVEL
        volumeMounts:
        - name: models-volume
          mountPath: /app/models
          readOnly: true
        - name: data-volume
          mountPath: /app/data
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 2
      volumes:
      - name: models-volume
        persistentVolumeClaim:
          claimName: models-pvc
      - name: data-volume
        persistentVolumeClaim:
          claimName: data-pvc

---
# kubernetes/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: models-pvc
  namespace: marine-detect
spec:
  accessModes:
    - ReadOnlyMany
  resources:
    requests:
      storage: 5Gi
  storageClassName: fast-ssd

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-pvc
  namespace: marine-detect
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  storageClassName: standard
```

#### 4. Service and Ingress

```yaml
# kubernetes/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: marine-detect-service
  namespace: marine-detect
  labels:
    app: marine-detect-api
spec:
  selector:
    app: marine-detect-api
  ports:
  - name: http
    protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP

---
# kubernetes/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: marine-detect-ingress
  namespace: marine-detect
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  tls:
  - hosts:
    - marine-detect.yourdomain.com
    secretName: marine-detect-tls
  rules:
  - host: marine-detect.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: marine-detect-service
            port:
              number: 80
```

#### 5. HorizontalPodAutoscaler

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: marine-detect-hpa
  namespace: marine-detect
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: marine-detect-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### Deploy to Kubernetes

```bash
# Apply all manifests
kubectl apply -f kubernetes/

# Check deployment status
kubectl get pods -n marine-detect
kubectl get services -n marine-detect
kubectl get ingress -n marine-detect

# View logs
kubectl logs -f deployment/marine-detect-api -n marine-detect

# Scale deployment
kubectl scale deployment marine-detect-api --replicas=5 -n marine-detect
```

## 📊 Monitoring and Observability

### Prometheus and Grafana

```yaml
# monitoring/prometheus-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: prometheus-config
  namespace: marine-detect
data:
  prometheus.yml: |
    global:
      scrape_interval: 15s
    scrape_configs:
    - job_name: 'marine-detect-api'
      static_configs:
      - targets: ['marine-detect-service:80']
      metrics_path: /metrics
      scrape_interval: 10s

---
# monitoring/grafana-dashboard.json
{
  "dashboard": {
    "title": "Marine Detect Monitoring",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "Requests/sec"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "http_request_duration_seconds",
            "legendFormat": "Response Time"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total{status=~\"5..\"}[5m])",
            "legendFormat": "Error Rate"
          }
        ]
      }
    ]
  }
}
```

### Application Metrics

```python
# marine_detect/metrics.py
from prometheus_client import Counter, Histogram, generate_latest
import time

# Metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
DETECTION_COUNT = Counter('detections_total', 'Total detections', ['species'])

def track_request(func):
    """Decorator to track request metrics."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            status = '200'
        except Exception as e:
            status = '500'
            raise
        finally:
            REQUEST_DURATION.observe(time.time() - start_time)
            REQUEST_COUNT.labels(method='POST', endpoint='/detect', status=status).inc()
        return result
    return wrapper

# Add to FastAPI app
from fastapi import FastAPI
from fastapi.responses import Response

app = FastAPI()

@app.get('/metrics')
async def metrics():
    return Response(generate_latest(), media_type='text/plain')
```

## 🔒 Security Considerations

### SSL/TLS Configuration

```nginx
# nginx-ssl.conf
server {
    listen 443 ssl http2;
    server_name marine-detect.yourdomain.com;
    
    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header Referrer-Policy "strict-origin-when-cross-origin";
    
    location / {
        proxy_pass http://marine_detect;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Environment Security

```bash
# .env.production
# Use strong, unique values in production
SECRET_KEY=your-very-long-random-secret-key
DATABASE_URL=postgresql://user:password@host:5432/db
REDIS_URL=redis://host:6379/0

# API security
API_KEY_REQUIRED=true
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=3600

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/marine-detect/app.log

# Performance
MAX_WORKERS=4
WORKER_TIMEOUT=300
MAX_REQUEST_SIZE=10485760  # 10MB
```

## 🔄 CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]
    tags: ['v*']

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
    - name: Checkout
      uses: actions/checkout@v3

    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2

    - name: Log in to Container Registry
      uses: docker/login-action@v2
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}

    - name: Build and push Docker image
      uses: docker/build-push-action@v4
      with:
        context: .
        platforms: linux/amd64,linux/arm64
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max

    - name: Deploy to production
      if: github.ref == 'refs/heads/main'
      run: |
        # Deploy script here
        echo "Deploying to production..."
        # kubectl apply -f kubernetes/
        # or docker-compose up -d
```

## 📞 Support

For deployment-related questions:
- Review the [troubleshooting guide](troubleshooting.md)
- Check [GitHub Issues](https://github.com/adityagit94/marine-detect/issues)
- Contact: [aditya_2312res46@iitp.ac.in](mailto:aditya_2312res46@iitp.ac.in)

---

**Next**: [Performance Guide](performance.md) | [Monitoring Guide](monitoring.md)
