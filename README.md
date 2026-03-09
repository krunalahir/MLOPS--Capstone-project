Capstone-Project
==============================

This is an end-to-end MLOps implementation for a text classification/ML pipeline, built on the Cookiecutter Data Science template.

    Core Components

    ML Pipeline (DVC-managed):
     1. Data Ingestion → Raw data collection
     2. Data Preprocessing → Cleaned interim data
     3. Feature Engineering → Processed features + vectorizer (NLP)
     4. Model Building → Trained model
     5. Model Evaluation → Metrics & experiment tracking
     6. Model Registration → Model registry integration


    Tech Stack:
     - ML Framework: scikit-learn, NLTK (NLP processing)
     - Experiment Tracking: MLflow
     - Data Versioning: DVC (with S3 storage)
     - Orchestration: DVC pipelines
     - Deployment: Docker + Kubernetes (EKS)
     - Container Registry: AWS ECR
     - Task Queue: Celery
     - Web Framework: Flask (with Gunicorn)


    Infrastructure:
     - Kubernetes deployment with 2 replicas
     - LoadBalancer service exposing port 5000
     - AWS integration (S3, ECR)
     - Secrets management via Kubernetes Secrets


    Key Features:
     - Automated ML pipeline with stage dependencies
     - Model versioning and registration
     - Metrics tracking (reports/metrics.json)
     - Production-ready Flask API deployment
     - Reproducible environment via requirements.txt
     - Makefile for common operations (data sync, linting, testing)


    Current Configuration:
     - Test split: 25%
     - Max features: 50 (TF-IDF/count vectorization)

    This project demonstrates production-grade MLOps practices covering the full lifecycle from data ingestion to deployed model serving.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
