# Project Structure Documentation

This document provides a comprehensive breakdown of the MLOps Capstone Project directory structure, explaining the purpose and contents of each file and directory.

---

## 📁 Root Directory Structure

```
MLOPs-Capstone-Project/
├── Configuration Files
├── Source Code (src/)
├── Data Directory (data/)
├── Models (models/)
├── Reports (reports/)
├── Flask Application (flask_app/)
├── Deployment (Dockerfile, deployment.yaml)
├── CI/CD (.github/)
├── Documentation (docs/)
├── Notebooks (notebooks/)
├── References (references/)
├── Tests (tests/)
└── Scripts (scripts/)
```

---

## 📄 Configuration Files

### `params.yaml`
**Purpose**: Centralized configuration for pipeline parameters.

**Contents**:
```yaml
data_ingestion:
  test_size: 0.25  # Train-test split ratio for data ingestion

feature_engineering:
  max_features: 50  # Maximum features for TF-IDF/Count vectorization
```

**Usage**: Parameters are automatically loaded by DVC stages during pipeline execution.

---

### `dvc.yaml`
**Purpose**: Defines the ML pipeline stages, dependencies, and outputs.

**Structure**:
```yaml
stages:
  data_ingestion:
    cmd: python src/datas/data_ingestion.py
    deps: [src/datas/data_ingestion.py]
    params: [data_ingestion.test_size]
    outs: [data/raw]

  data_preprocessing:
    cmd: python src/datas/data_preprocessor.py
    deps: [data/raw, src/datas/data_preprocessor.py]
    outs: [data/interim]

  feature_engineering:
    cmd: python src/features/feature_engineering.py
    deps: [data/interim, src/features/feature_engineering.py]
    params: [feature_engineering.max_features]
    outs: [data/processed, models/vectorizer.pkl]

  model_building:
    cmd: python src/model/model_building.py
    deps: [data/processed, src/model/model_building.py]
    outs: [models/model.pkl]

  model_evaluation:
    cmd: python src/model/model_evaluation.py
    deps: [models/model.pkl, src/model/model_evaluation.py]
    metrics: [reports/metrics.json]
    outs: [reports/experiment_info.json]

  model_registration:
    cmd: python src/model/register_model.py
    deps: [reports/experiment_info.json, src/model/register_model.py]
```

**Key Concepts**:
- `cmd`: Command to execute for the stage
- `deps`: Stage dependencies (files or directories)
- `params`: Parameters from `params.yaml`
- `outs`: Stage outputs (tracked by DVC)
- `metrics`: Metrics files (special output for tracking)

---

### `dvc.lock`
**Purpose**: Locks pipeline stage outputs to specific checksums for reproducibility.

**Auto-generated**: Do not edit manually. Updated automatically when running `dvc repro`.

---

### `.dvcignore`
**Purpose**: Specifies files and directories that DVC should ignore (similar to `.gitignore`).

**Typical Contents**:
```
# Ignore Python cache
__pycache__/
*.py[cod]
*$py.class

# Ignore virtual environments
venv/
.env/
capstoneenv/

# Ignore IDE files
.idea/
.vscode/

# Ignore system files
.DS_Store
```

---

### `requirements.txt`
**Purpose**: Lists all Python dependencies for reproducible environment setup.

**Key Dependencies**:
- **ML Libraries**: scikit-learn, nltk, pandas, numpy
- **MLOps Tools**: dvc, mlflow, mlflow-skinny
- **Web Framework**: Flask, gunicorn, waitress
- **AWS Integration**: boto3, botocore, s3fs
- **Containerization**: docker
- **Task Queue**: celery
- **Monitoring**: prometheus_client

**Usage**:
```bash
pip install -r requirements.txt
```

---

### `setup.py`
**Purpose**: Makes the project installable as a Python package, enabling imports from `src`.

**Contents**:
```python
from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='this repo contains end to end mlops implementation',
    author='Krunal Bhammar',
    license='',
)
```

**Usage**:
```bash
# Install in development mode
pip install -e .

# Now you can import from src
from src.datas import data_ingestion
```

---

### `Makefile`
**Purpose**: Automates common development and deployment tasks.

**Available Commands**:

| Command | Description |
|---------|-------------|
| `make help` | Show all available commands |
| `make requirements` | Install Python dependencies |
| `make data` | Generate dataset (legacy) |
| `make lint` | Run flake8 code linting |
| `make sync_data_to_s3` | Upload data to S3 bucket |
| `make sync_data_from_s3` | Download data from S3 bucket |
| `make clean` | Remove compiled Python files |
| `make create_environment` | Create virtual environment |
| `make test_environment` | Verify environment setup |

**Usage Examples**:
```bash
# Install dependencies
make requirements

# Sync data to S3
make sync_data_to_s3 BUCKET=my-bucket-name

# Clean compiled files
make clean
```

---

### `tox.ini`
**Purpose**: Configures Tox for automated testing across multiple environments.

**Typical Configuration**:
```ini
[tox]
envlist = py310

[testenv]
deps = -rrequirements.txt
commands = pytest tests/
```

**Usage**:
```bash
# Run tests in isolated environment
tox
```

---

### `.gitignore`
**Purpose**: Specifies files and directories that Git should ignore.

**Key Ignored Patterns**:
- Python cache files (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`, `capstoneenv/`)
- Data directories (`data/`, `models/`)
- IDE settings (`.idea/`)
- DVC cache (`.dvc/`)
- Credentials (`cred.txt`, `.pypirc`)

---

### `test_environment.py`
**Purpose**: Verifies that the development environment is correctly configured.

**Checks**:
- Python version
- Required packages installation
- Directory structure existence
- Environment variables

**Usage**:
```bash
python test_environment.py
```

---

## 📂 Source Code Directory (`src/`)

### `src/__init__.py`
**Purpose**: Makes `src` a Python package, enabling imports.

---

### `src/datas/` - Data Ingestion and Preprocessing

#### `src/datas/data_ingestion.py`
**Purpose**: Collects and stores raw data from external sources.

**Responsibilities**:
- Download data from APIs, databases, or files
- Validate data integrity
- Save raw data to `data/raw/`
- Log ingestion statistics

**Input**: External data sources
**Output**: `data/raw/`

---

#### `src/datas/data_preprocessor.py`
**Purpose**: Cleans and normalizes raw text data.

**Responsibilities**:
- Text normalization (lowercase, punctuation removal)
- Stopword removal (NLTK)
- Lemmatization/stemming
- Tokenization
- Handle missing values

**Input**: `data/raw/`
**Output**: `data/interim/`

**Example Processing**:
```python
# Raw: "The Quick Brown Fox!!!"
# Preprocessed: "quick brown fox"
```

---

### `src/features/` - Feature Engineering

#### `src/features/feature_engineering.py`
**Purpose**: Transforms preprocessed text into numerical features.

**Responsibilities**:
- TF-IDF vectorization
- Count vectorization
- N-gram extraction
- Feature selection
- Save vectorizer for inference

**Input**: `data/interim/`
**Output**: `data/processed/`, `models/vectorizer.pkl`

**Configuration**:
```yaml
feature_engineering:
  max_features: 50  # Maximum number of features
```

---

### `src/model/` - Model Training and Evaluation

#### `src/model/model_building.py`
**Purpose**: Trains machine learning classification models.

**Responsibilities**:
- Load processed data
- Split into train/test sets
- Train classifier (e.g., Logistic Regression, Naive Bayes)
- Save trained model
- Log training parameters

**Input**: `data/processed/`
**Output**: `models/model.pkl`

**Typical Models**:
- Logistic Regression
- Multinomial Naive Bayes
- Support Vector Machines
- Random Forest

---

#### `src/model/model_evaluation.py`
**Purpose**: Evaluates model performance and tracks metrics.

**Responsibilities**:
- Load trained model
- Evaluate on test set
- Calculate metrics (accuracy, precision, recall, F1)
- Generate confusion matrix
- Log metrics to MLflow
- Save metrics to JSON

**Input**: `models/model.pkl`
**Output**: `reports/metrics.json`, `reports/experiment_info.json`

**Metrics Tracked**:
```json
{
  "accuracy": 0.92,
  "precision": 0.91,
  "recall": 0.90,
  "f1_score": 0.905
}
```

---

#### `src/model/register_model.py`
**Purpose**: Registers trained models in MLflow Model Registry.

**Responsibilities**:
- Load model and metrics
- Connect to MLflow tracking server
- Register model with version
- Add metadata (metrics, parameters)
- Set model stage (Staging, Production)

**Input**: `reports/experiment_info.json`
**Output**: MLflow Model Registry entry

---

### `src/connections/` - External Connections

**Purpose**: Manages connections to external services (databases, APIs, cloud services).

**Typical Contents**:
- Database connection strings
- AWS S3 client configuration
- MLflow tracking URI setup
- API authentication handlers

**Example**:
```python
import boto3
from botocore.exceptions import NoCredentialsError

def get_s3_client():
    return boto3.client('s3')
```

---

### `src/logger/` - Logging Utilities

**Purpose**: Centralized logging configuration for the project.

**Typical Contents**:
```python
import logging

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Add handlers and formatters
    return logger
```

**Usage**:
```python
from src.logger import get_logger
logger = get_logger(__name__)
logger.info("Pipeline stage completed")
```

---

### `src/visualization/` - Visualization Scripts

**Purpose**: Creates visualizations for data exploration and results reporting.

**Typical Scripts**:
- Data distribution plots
- Confusion matrix heatmaps
- ROC curves
- Feature importance charts

**Output**: `reports/figures/`

---

## 📂 Data Directory (`data/`)

Managed by DVC. Contains three subdirectories representing different stages of data processing:

### `data/raw/`
**Purpose**: Original, immutable data dump.

**Characteristics**:
- Never modify files in this directory
- Source of truth for all downstream processing
- Typically large files tracked by DVC
- Examples: `train.csv`, `test.csv`, `raw_text.txt`

---

### `data/interim/`
**Purpose**: Intermediate data that has been transformed.

**Characteristics**:
- Cleaned and preprocessed data
- May require additional processing before modeling
- Examples: `cleaned_train.csv`, `tokenized_text.pkl`

---

### `data/processed/`
**Purpose**: Final, canonical data sets for modeling.

**Characteristics**:
- Ready for feature engineering or modeling
- Often in vectorized format
- Examples: `train_features.npy`, `test_labels.csv`

---

## 📂 Models Directory (`models/`)

**Purpose**: Stores trained models, vectorizers, and model artifacts.

### Key Files:

#### `models/model.pkl`
- Trained and serialized scikit-learn model
- Used for inference in production
- Versioned by DVC

#### `models/vectorizer.pkl`
- Fitted TF-IDF or Count vectorizer
- Required for transforming new text data
- Must match the model's training preprocessing

---

## 📂 Reports Directory (`reports/`)

### `reports/metrics.json`
**Purpose**: Stores model evaluation metrics in JSON format.

**Structure**:
```json
{
  "accuracy": 0.92,
  "precision": 0.91,
  "recall": 0.90,
  "f1_score": 0.905,
  "confusion_matrix": [[45, 5], [3, 47]]
}
```

---

### `reports/experiment_info.json`
**Purpose**: Comprehensive experiment tracking data.

**Contents**:
- Model parameters
- Training metrics
- Timestamp
- Git commit hash
- Environment details

---

### `reports/figures/`
**Purpose**: Generated graphics and figures for reporting.

**Typical Files**:
- `confusion_matrix.png`
- `roc_curve.png`
- `feature_importance.png`
- `data_distribution.png`

---

## 📂 Flask Application (`flask_app/`)

### `flask_app/app.py`
**Purpose**: Production Flask REST API for model inference.

**Features**:
- `/predict` endpoint for text classification
- Model loading from pickle files
- Input validation
- Error handling
- Prometheus metrics endpoint
- Gunicorn WSGI server integration

**Example Request**:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This is a sample text"}'
```

**Example Response**:
```json
{
  "prediction": "positive",
  "confidence": 0.87,
  "probabilities": {
    "positive": 0.87,
    "negative": 0.13
  }
}
```

---

### `flask_app/app_local.py`
**Purpose**: Local development version of Flask app.

**Differences from `app.py`**:
- Debug mode enabled
- Local model paths
- Simplified error handling
- No Prometheus metrics

---

### `flask_app/preprocessing_utility.py`
**Purpose**: Preprocessing functions for inference.

**Responsibilities**:
- Load vectorizer
- Apply same preprocessing as training
- Transform text to features
- Ensure consistency between training and inference

**Functions**:
```python
def load_vectorizer():
    """Load saved vectorizer from disk"""

def preprocess_text(text):
    """Apply preprocessing to input text"""

def predict(text):
    """Full prediction pipeline: preprocess → vectorize → predict"""
```

---

### `flask_app/load_model_test.py`
**Purpose**: Tests for model loading and inference.

**Test Cases**:
- Model file exists
- Model loads successfully
- Vectorizer loads successfully
- Prediction returns valid output

---

### `flask_app/requirements.txt`
**Purpose**: Flask app-specific dependencies.

**Note**: May be a subset of root `requirements.txt`.

---

### `flask_app/templates/`
**Purpose**: HTML templates for web interface (if applicable).

**Typical Files**:
- `index.html` - Main page
- `predict.html` - Prediction form

---

## 📂 Deployment Files

### `Dockerfile`
**Purpose**: Defines container image for deployment.

**Contents**:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY flask_app/ /app/
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

RUN pip install -r requirements.txt
RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
```

**Build Process**:
```bash
docker build -t flask-app:latest .
```

---

### `deployment.yaml`
**Purpose**: Kubernetes deployment and service configuration.

**Components**:

#### Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: flask-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: flask-app
  template:
    spec:
      containers:
      - name: flask-app
        image: 020866158197.dkr.ecr.us-east-1.amazonaws.com/flask-app:latest
        ports:
        - containerPort: 5000
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "1"
```

#### Service
```yaml
apiVersion: v1
kind: Service
metadata:
  name: flask-app-service
spec:
  type: LoadBalancer
  selector:
    app: flask-app
  ports:
    - protocol: TCP
      port: 5000
      targetPort: 5000
```

**Deploy**:
```bash
kubectl apply -f deployment.yaml
```

---

## 📂 Scripts Directory (`scripts/`)

### `scripts/promote_model.py`
**Purpose**: Automates model promotion between stages (e.g., Staging → Production).

**Responsibilities**:
- Connect to MLflow
- Get latest model version
- Validate performance metrics
- Update model stage
- Log promotion event

**Usage**:
```bash
python scripts/promote_model.py --model-name text-classifier --stage Production
```

---

## 📂 CI/CD Directory (`.github/`)

### `.github/workflows/`
**Purpose**: GitHub Actions workflow definitions.

**Typical Workflows**:

#### `ci.yml` - Continuous Integration
```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest tests/
      - name: Lint
        run: make lint
```

#### `cd.yml` - Continuous Deployment
```yaml
name: CD
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Build and push Docker image
      - name: Deploy to Kubernetes
```

---

## 📂 Documentation Directory (`docs/`)

**Purpose**: Sphinx documentation source files.

**Typical Structure**:
```
docs/
├── conf.py              # Sphinx configuration
├── index.rst            # Main documentation file
├── installation.rst     # Installation guide
├── usage.rst            # Usage documentation
├── api.rst              # API reference
└── _build/              # Generated documentation (ignored)
```

**Build Documentation**:
```bash
cd docs
make html
# Open _build/html/index.html
```

---

## 📂 Notebooks Directory (`notebooks/`)

**Purpose**: Jupyter notebooks for exploratory data analysis and prototyping.

**Naming Convention**:
```
<order>.<author>-<description>.ipynb

Examples:
1.0-kb-initial-data-exploration.ipynb
2.0-kb-preprocessing-experiments.ipynb
3.0-kb-model-comparison.ipynb
```

**Best Practices**:
- Number notebooks for chronological order
- Include author initials
- Use descriptive names
- Document findings and conclusions

---

## 📂 References Directory (`references/`)

**Purpose**: Data dictionaries, manuals, and explanatory materials.

**Typical Contents**:
- Data dictionary (column descriptions)
- Domain documentation
- API documentation
- Research papers
- Manual data collection guides

---

## 📂 Tests Directory (`tests/`)

**Purpose**: Unit tests, integration tests, and end-to-end tests.

**Typical Structure**:
```
tests/
├── __init__.py
├── test_data_ingestion.py
├── test_preprocessing.py
├── test_feature_engineering.py
├── test_model_building.py
├── test_model_evaluation.py
└── test_flask_app.py
```

**Run Tests**:
```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_model_building.py -v
```

---

## 📂 Logs Directory (`logs/`)

**Purpose**: Application and pipeline logs.

**Typical Files**:
- `pipeline.log` - DVC pipeline execution logs
- `app.log` - Flask application logs
- `mlflow.log` - MLflow tracking logs

**Note**: Usually ignored by Git (`.gitignore`).

---

## 📄 Other Important Files

### `LICENSE`
**Purpose**: Project license file (MIT, Apache, etc.).

---

### `README.md`
**Purpose**: Main project documentation (this file's companion).

---

### `PROJECT_STRUCTURE.md`
**Purpose**: This file - detailed structure documentation.

---

## 🗺️ File Flow Diagram

```
Data Flow Through Pipeline:
┌─────────────┐
│ data/raw/   │ ← Data Ingestion (src/datas/data_ingestion.py)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│data/interim/│ ← Preprocessing (src/datas/data_preprocessor.py)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│data/process/│ ← Feature Engineering (src/features/feature_engineering.py)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│models/      │ ← Model Building (src/model/model_building.py)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│reports/     │ ← Model Evaluation (src/model/model_evaluation.py)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│MLflow       │ ← Model Registration (src/model/register_model.py)
└─────────────┘
```

---

## 📝 Summary

| Directory | Purpose | Managed By |
|-----------|---------|------------|
| `src/` | Source code | Git |
| `data/` | Data artifacts | DVC |
| `models/` | Model artifacts | DVC |
| `reports/` | Metrics and reports | Git/DVC |
| `flask_app/` | API application | Git |
| `tests/` | Test suite | Git |
| `docs/` | Documentation | Git |
| `notebooks/` | Exploratory analysis | Git |
| `.github/` | CI/CD workflows | Git |

---

<p align="center">
  <strong>For more information, see the main <a href="README.md">README.md</a></strong>
</p>
