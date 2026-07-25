# MLOps Capstone Project: End-to-End Sentiment Analysis

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![DVC](https://img.shields.io/badge/DVC-3.53.0-green.svg)](https://dvc.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.15.0-orange.svg)](https://mlflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.5.1-red.svg)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![Kubernetes](https://img.shields.io/badge/kubernetes-deployment-blue.svg)](https://kubernetes.io/)

A production-grade **MLOps implementation** for text classification, demonstrating industry best practices across the entire machine learning lifecycle—from data ingestion to deployed model serving on Kubernetes.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [ML Pipeline Stages](#ml-pipeline-stages)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Deployment](#deployment)
- [Monitoring & Experiment Tracking](#monitoring--experiment-tracking)
- [Testing](#testing)
- [CI/CD](#cicd)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## 🎯 Overview

This project implements a **complete MLOps pipeline** for text classification using a Cookiecutter Data Science template structure. It showcases modern DevOps practices applied to machine learning, including:

- **Automated ML pipelines** with DVC (Data Version Control)
- **Experiment tracking** and model registry with MLflow
- **Containerization** with Docker
- **Orchestration** with Kubernetes on AWS EKS
- **Model serving** via RESTful Flask API
- **Infrastructure as Code** with Kubernetes manifests

The pipeline processes raw text data through preprocessing, feature engineering, model training, evaluation, and registration, culminating in a production-ready deployment on AWS infrastructure.

---

## ✨ Features

### Core Capabilities

- **🔄 Automated ML Pipeline**: Six-stage pipeline with automatic dependency management
- **📊 Data Versioning**: Full lineage tracking with DVC and S3 backend
- **🔬 Experiment Tracking**: Comprehensive metrics, parameters, and artifacts logging
- **📦 Model Registry**: Versioned model storage and promotion workflows
- **🚀 Production Deployment**: Kubernetes-ready with load balancing and auto-scaling
- **🔐 Secrets Management**: Secure credential handling via Kubernetes Secrets

### Technical Highlights

- **NLP Processing**: NLTK-based text preprocessing with stopword removal and lemmatization
- **Feature Engineering**: TF-IDF/Count vectorization with configurable parameters
- **Model Training**: Scikit-learn classifiers with hyperparameter flexibility
- **Evaluation Metrics**: Accuracy, precision, recall, F1-score, and confusion matrix
- **API Serving**: Flask REST API with Gunicorn WSGI server
- **Container Optimization**: Multi-stage Docker builds for minimal image size

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MLOps Pipeline Architecture                       │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Data        │────▶│  Data        │────▶│  Feature     │
│  Ingestion   │     │  Preprocessing│    │  Engineering │
└──────────────┘     └──────────────┘     └──────────────┘
         │                   │                      │
         ▼                   ▼                      ▼
    S3 Storage         Cleaned Data          Vectorizers
    (DVC)              (Interim)             (Models/)

┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Model       │────▶│  Model       │────▶│  Model       │
│  Building    │     │  Evaluation  │     │  Registration│
└──────────────┘     └──────────────┘     └──────────────┘
         │                   │                      │
         ▼                   ▼                      ▼
    Trained Model      Metrics (MLflow)      Model Registry
    (models/)          Reports/              (S3/MLflow)
```

### Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  AWS Cloud Infrastructure               │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  |            Kubernetes Cluster (EKS)              │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │   Flask App Deployment (2 replicas)     │    │   │
│  │  │   ┌─────────┐  ┌─────────┐             │    │   │
│  │  │   │  Pod 1  │  │  Pod 2  │             │    │   │
│  │  │   └────┬────┘  └────┬────┘             │    │   │
│  │  └────────┼────────────┼──────────────────┘    │   │
│  │           └────────────┴──────────┐            │   │
│  │                                   ▼            │   │
│  │                        ┌─────────────────┐     │   │
│  │                        │   LoadBalancer  │     │   │
│  │                        │   Service:5000  │     │   │
│  │                        └─────────────────┘     │   │
│  └─────────────────────────────────────────────────┘   │
│                         ▲                               │
│                         │                               │
│  ┌──────────────────────┴────────────────────────┐     │
│  │  AWS ECR (Container Registry)                 │     │
│  │  ┌─────────────────────────────────────────┐  │     │
│  │  │  flask-app:latest                       │  │     │
│  │  └─────────────────────────────────────────┘  │     │
│  └─────────────────────────────────────────────────┘     │
│                         ▲                               │
│  ┌──────────────────────┴────────────────────────┐     │
│  │  AWS S3 (DVC Remote Storage)                  │     │
│  │  ┌─────────────────────────────────────────┐  │     │
│  │  │  Models | Data | Artifacts              │  │     │
│  │  └─────────────────────────────────────────┘  │     │
│  └─────────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```


---

## 🔄 ML Pipeline Stages

The pipeline consists of **six sequential stages**, each with defined inputs, outputs, and dependencies:

| Stage | Description | Input | Output |
|-------|-------------|-------|--------|
| **1. Data Ingestion** | Collect and store raw data | External sources | `data/raw/` |
| **2. Data Preprocessing** | Clean and normalize text | `data/raw/` | `data/interim/` |
| **3. Feature Engineering** | Transform text to features | `data/interim/` | `data/processed/`, `models/vectorizer.pkl` |
| **4. Model Building** | Train classification model | `data/processed/` | `models/model.pkl` |
| **5. Model Evaluation** | Evaluate and track metrics | `models/model.pkl` | `reports/metrics.json`, `reports/experiment_info.json` |
| **6. Model Registration** | Register model for deployment | `reports/experiment_info.json` | MLflow Model Registry |

### Pipeline Graph

```bash
dvc dag
```

```
data_ingestion → data_preprocessing → feature_engineering → model_building → model_evaluation → model_registration
```

---

## 🛠️ Tech Stack

### Machine Learning & Data Processing

| Technology | Version | Purpose |
|------------|---------|---------|
| **Python** | 3.10+ | Core language |
| **Scikit-Learn** | 1.5.1 | ML models and preprocessing |
| **NLTK** | 3.8.1 | Natural language processing |
| **Pandas** | 2.2.2 | Data manipulation |
| **NumPy** | 1.26.4 | Numerical computing |
| **PyArrow** | 15.0.2 | Data serialization |

### MLOps & Experiment Tracking

| Technology | Version | Purpose |
|------------|---------|---------|
| **DVC** | 3.53.0 | Data versioning and pipelines |
| **MLflow** | 2.15.0 | Experiment tracking and model registry |
| **Git** | - | Version control |
| **GitHub Actions** | - | CI/CD automation |

### Deployment & Infrastructure

| Technology | Version | Purpose |
|------------|---------|---------|
| **Docker** | - | Containerization |
| **Kubernetes** | - | Container orchestration |
| **AWS EKS** | - | Managed Kubernetes |
| **AWS ECR** | - | Container registry |
| **AWS S3** | - | Artifact storage |
| **Flask** | 3.0.3 | REST API framework |
| **Gunicorn** | - | WSGI HTTP server |

### Development & Quality

| Technology | Version | Purpose |
|------------|---------|---------|
| **Tox** | - | Test automation |
| **Flake8** | - | Code linting |
| **Pytest** | - | Unit testing |
| **Make** | - | Command automation |

---

## 📁 Project Structure

```
MLOPs-Capstone-Project/
│
├── 📄 Configuration Files
│   ├── params.yaml              # Pipeline parameters (test_size, max_features)
│   ├── dvc.yaml                 # DVC pipeline definition
│   ├── dvc.lock                 # DVC pipeline lock file
│   ├── .dvcignore               # DVC ignore patterns
│   ├── requirements.txt         # Python dependencies
│   ├── setup.py                 # Package setup configuration
│   ├── tox.ini                  # Tox test configuration
│   └── Makefile                 # Makefile commands
│
├── 📂 Source Code
│   └── src/
│       ├── __init__.py
│       ├── datas/               # Data ingestion and preprocessing
│       │   ├── data_ingestion.py
│       │   └── data_preprocessor.py
│       ├── features/            # Feature engineering
│       │   └── feature_engineering.py
│       ├── model/               # Model training and evaluation
│       │   ├── model_building.py
│       │   ├── model_evaluation.py
│       │   └── register_model.py
│       ├── connections/         # Database and service connections
│       ├── logger/              # Logging utilities
│       └── visualization/       # Visualization scripts
│
├── 📂 Data Directory (DVC-managed)
│   └── data/
│       ├── raw/                 # Original immutable data
│       ├── interim/             # Cleaned and preprocessed data
│       └── processed/           # Final feature datasets
│
├── 📂 Models
│   └── models/
│       ├── model.pkl            # Trained model artifact
│       └── vectorizer.pkl       # Feature vectorizer
│
├── 📂 Reports & Metrics
│   └── reports/
│       ├── metrics.json         # Model evaluation metrics
│       ├── experiment_info.json # Experiment tracking data
│       └── figures/             # Generated visualizations
│
├── 📂 Flask Application
│   └── flask_app/
│       ├── app.py               # Production Flask API
│       ├── app_local.py         # Local development Flask app
│       ├── preprocessing_utility.py  # Preprocessing utilities
│       ├── load_model_test.py   # Model loading tests
│       └── templates/           # HTML templates
│
├── 📂 Deployment
│   ├── Dockerfile               # Container image definition
│   ├── deployment.yaml          # Kubernetes deployment manifest
│   └── scripts/
│       └── promote_model.py     # Model promotion script
│
├── 📂 CI/CD
│   └── .github/
│       └── workflows/           # GitHub Actions workflows
│
├── 📂 Documentation
│   └── docs/                    # Sphinx documentation
│
├── 📂 Notebooks
│   └── notebooks/               # Jupyter notebooks for exploration
│
├── 📂 References
│   └── references/              # Data dictionaries and documentation
│
├── 📂 Tests
│   └── tests/                   # Unit and integration tests
│
└── 📄 Other Files
    ├── README.md                # This file
    ├── PROJECT_STRUCTURE.md     # Detailed structure documentation
    ├── LICENSE                  # Project license
    ├── .gitignore               # Git ignore patterns
    └── test_environment.py      # Environment setup verification
```

For a detailed breakdown of each component, see [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md).

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+**
- **Git**
- **DVC** (for data versioning)
- **Docker** (for containerization)
- **AWS CLI** (for S3/ECR integration)
- **kubectl** (for Kubernetes deployment)

### 5-Minute Setup

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/MLOPs-Capstone-Project.git
cd MLOPs-Capstone-Project

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK data
python -m nltk.downloader stopwords wordnet

# 5. Pull data artifacts (requires DVC setup)
dvc pull

# 6. Run the full pipeline
dvc repro

# 7. Start the Flask API locally
cd flask_app
python app_local.py
```

---

## 📥 Installation

### Development Environment

```bash
# Clone repository
git clone https://github.com/yourusername/MLOPs-Capstone-Project.git
cd MLOPs-Capstone-Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install package in development mode
pip install -e .

# Install development dependencies
pip install -r requirements.txt

# Verify environment setup
python test_environment.py
```

### DVC Configuration

```bash
# Initialize DVC (if not already initialized)
dvc init

# Configure S3 remote (update with your bucket)
dvc remote add -d s3://your-bucket-name/dvc-storage

# Configure AWS credentials
aws configure

# Pull data artifacts
dvc pull
```

### MLflow Configuration

```bash
# Set MLflow tracking URI
export MLFLOW_TRACKING_URI=http://localhost:5000

# Or for production
export MLFLOW_TRACKING_URI=https://your-mlflow-server.com
```

---

## 💻 Usage

### Running the ML Pipeline

```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro data_ingestion
dvc repro data_preprocessing
dvc repro feature_engineering
dvc repro model_building
dvc repro model_evaluation
dvc repro model_registration

# View pipeline DAG
dvc dag

# Check pipeline status
dvc status
```

### Using Makefile Commands

```bash
# Show all available commands
make help

# Install dependencies
make requirements

# Run linting
make lint

# Sync data to S3
make sync_data_to_s3 BUCKET=your-bucket-name

# Sync data from S3
make sync_data_from_s3 BUCKET=your-bucket-name

# Clean compiled files
make clean
```

### Running the Flask API

```bash
# Local development
cd flask_app
python app_local.py

# Production with Gunicorn (inside Docker)
gunicorn --bind 0.0.0.0:5000 --timeout 120 app:app

# Test API endpoint
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here"}'
```

### Docker Commands

```bash
# Build Docker image
docker build -t flask-app:latest .

# Run container locally
docker run -p 5000:5000 flask-app:latest

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 020866158197.dkr.ecr.us-east-1.amazonaws.com
docker tag flask-app:latest 020866158197.dkr.ecr.us-east-1.amazonaws.com/flask-app:latest
docker push 020866158197.dkr.ecr.us-east-1.amazonaws.com/flask-app:latest
```

### Kubernetes Deployment

```bash
# Deploy to Kubernetes
kubectl apply -f deployment.yaml

# Check deployment status
kubectl get deployments
kubectl get pods
kubectl get services

# Access service
kubectl port-forward service/flask-app-service 5000:5000

# View logs
kubectl logs -l app=flask-app

# Scale deployment
kubectl scale deployment flask-app --replicas=3

# Rollback deployment
kubectl rollout undo deployment/flask-app
```

---

## ⚙️ Configuration

### Pipeline Parameters (`params.yaml`)

```yaml
data_ingestion:
  test_size: 0.25  # Train-test split ratio

feature_engineering:
  max_features: 50  # Maximum features for vectorization
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MLFLOW_TRACKING_URI` | MLflow server URI | `http://localhost:5000` |
| `AWS_PROFILE` | AWS credentials profile | `default` |
| `DVC_REMOTE` | DVC remote storage name | `s3` |
| `CAPSTONE_TEST` | Application secret key | (from Kubernetes Secret) |

### DVC Configuration (`.dvc/config`)

```ini
[core]
    remote = s3
['remote "s3"']
    url = s3://your-bucket-name/dvc-storage
    region = us-east-1
```

---

## 🚢 Deployment

### Docker Deployment

```bash
# Build image
docker build -t flask-app:latest .

# Test locally
docker run -p 5000:5000 flask-app:latest

# Push to ECR
docker push 020866158197.dkr.ecr.us-east-1.amazonaws.com/flask-app:latest
```

### Kubernetes Deployment

```bash
# Create Kubernetes secrets
kubectl create secret generic capstone-secret \
  --from-literal=CAPSTONE_TEST=your-secret-key

# Apply deployment
kubectl apply -f deployment.yaml

# Verify deployment
kubectl get all -l app=flask-app
```

### Deployment Specifications

| Resource | Specification |
|----------|---------------|
| **Replicas** | 2 |
| **CPU Request** | 250m |
| **CPU Limit** | 1000m |
| **Memory Request** | 256Mi |
| **Memory Limit** | 512Mi |
| **Port** | 5000 |
| **Service Type** | LoadBalancer |

---

## 📊 Monitoring & Experiment Tracking

### MLflow Integration

All experiments are tracked with MLflow:

```bash
# Start MLflow UI
mlflow ui

# Access UI at http://localhost:5000
```

### Tracked Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Precision score (weighted)
- **Recall**: Recall score (weighted)
- **F1-Score**: F1 score (weighted)
- **Confusion Matrix**: Classification confusion matrix

### Metrics Location

- `reports/metrics.json`: JSON format metrics
- `reports/experiment_info.json`: Full experiment details
- MLflow UI: Interactive visualizations and comparisons

```bash
# Access metrics endpoint
curl http://localhost:5000/metrics
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run tox environments
tox

# Test specific module
pytest tests/test_model_building.py -v
```

### Test Environment Verification

```bash
# Verify environment setup
python test_environment.py
```

---

## 🔄 CI/CD

### GitHub Actions Workflows

Located in `.github/workflows/`:

- **CI Pipeline**: Automated testing on pull requests
- **CD Pipeline**: Automated deployment on merge to main
- **DVC Pipeline**: Data versioning and model training

### Pipeline Stages

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│   Lint   │──▶│   Test   │──▶│   Build  │──▶│  Deploy  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
```

---

## 🤝 Contributing

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Make your changes**
4. **Run tests** (`pytest tests/`)
5. **Lint code** (`make lint`)
6. **Commit changes** (`git commit -m 'Add amazing feature'`)
7. **Push to branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request**

### Code Style

- Follow **PEP 8** guidelines
- Use **type hints** where applicable
- Write **docstrings** for public functions
- Maintain **test coverage** above 80%

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: Add new feature
fix: Fix bug
docs: Update documentation
style: Format code
refactor: Refactor code
test: Add tests
chore: Update dependencies
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 📞 Contact

**Author**: Krunal Bhammar

**Project**: MLOps Capstone Project

**Repository**: [GitHub](https://github.com/yourusername/MLOPs-Capstone-Project)

---

## 🙏 Acknowledgments

- [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) template
- [DVC](https://dvc.org/) for data versioning
- [MLflow](https://mlflow.org/) for experiment tracking
- [Scikit-Learn](https://scikit-learn.org/) for machine learning
- [Flask](https://flask.palletsprojects.com/) for web framework

---

## 📚 Additional Resources

- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Detailed project structure documentation
- [docs/](docs/) - Sphinx documentation
- [notebooks/](notebooks/) - Jupyter notebooks for exploration
- [references/](references/) - Data dictionaries and manuals

---

<p align="center">
  <strong>Built with ❤️ using MLOps best practices</strong>
</p>
