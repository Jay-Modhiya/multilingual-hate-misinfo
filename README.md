# 🛡️ Multilingual Hate Speech & Misinformation Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-EE4C2C?style=for-the-badge&logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=for-the-badge&logo=streamlit)
![MLflow](https://img.shields.io/badge/MLflow-3.11.1-0194E2?style=for-the-badge&logo=mlflow)
![Docker](https://img.shields.io/badge/Docker-Containerised-2496ED?style=for-the-badge&logo=docker)
![AWS](https://img.shields.io/badge/AWS-EC2%20Deployed-FF9900?style=for-the-badge&logo=amazonaws)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF?style=for-the-badge&logo=githubactions)
![DVC](https://img.shields.io/badge/DVC-Data%20Versioned-945DD6?style=for-the-badge&logo=dvc)

**A production-grade MLOps project for multilingual hate speech and misinformation detection in English, Hindi, and Hinglish using the MuRIL transformer.**

*MTech AI & ML — CA-4 MLOps Mini Project | Symbiosis Institute of Technology, Pune*

**Jay Modhiya (25070149031) | Krishna Nandi (25070149011)**

[🔴 Live Demo](http://3.110.197.48:8501) · [📊 MLflow UI](#-step-6-view-mlflow-experiments) · [📋 Project Report](#-project-report)

</div>

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Model Results](#-model-results)
3. [Architecture](#-system-architecture)
4. [Project Structure](#-project-structure)
5. [Datasets](#-datasets)
6. [Quick Start — Full Setup Guide](#-quick-start--full-setup-guide)
   - [Step 1 — Prerequisites](#step-1--prerequisites)
   - [Step 2 — Clone & Virtual Environment](#step-2--clone--virtual-environment)
   - [Step 3 — Install Dependencies](#step-3--install-dependencies)
   - [Step 4 — Setup HASOC Dataset (DVC)](#step-4--setup-hasoc-dataset-dvc)
   - [Step 5 — Run the Streamlit App](#step-5--run-the-streamlit-app)
   - [Step 6 — View MLflow Experiments](#step-6--view-mlflow-experiments)
   - [Step 7 — Run DVC Pipeline](#step-7--run-dvc-pipeline)
   - [Step 8 — Run Unit Tests](#step-8--run-unit-tests)
   - [Step 9 — Build Docker Image](#step-9--build-docker-image)
7. [MLOps Stack](#-mlops-stack)
8. [Streamlit App Pages](#-streamlit-app-pages)
9. [Batch Deployment Strategy](#-batch-deployment-strategy)
10. [Bias & Fairness Analysis](#-bias--fairness-analysis)
11. [Data Drift Monitoring](#-data-drift-monitoring)
12. [CI/CD Pipeline](#-cicd-pipeline)
13. [AWS Deployment](#-aws-deployment)
14. [Viva — Key Talking Points](#-viva--key-talking-points)

---

## 🎯 Project Overview

This project builds a **multilingual content moderation system** that detects hate speech and misinformation in English, Hindi, and Hinglish text. It is built on **google/muril-base-cased** — a 238M parameter transformer model pre-trained by Google specifically on 17 Indian languages, making it ideal for Indian social media content moderation.

The project implements the **full MLOps lifecycle**:

| Component | Tool | Purpose |
|-----------|------|---------|
| Model | MuRIL (google/muril-base-cased) | Multilingual Indian language NLP |
| Data Versioning | DVC | Track & reproduce datasets |
| Experiment Tracking | MLflow | Log metrics, artifacts, models |
| Containerisation | Docker + AWS ECR | Portable deployment |
| CI/CD | GitHub Actions (3 stages) | Automated test → build → deploy |
| Cloud Deployment | AWS EC2 (t2.micro) | Live web application |
| Explainability | LIME | Word-level feature importance |
| Bias Analysis | Equalized Odds | Fairness across language groups |
| Drift Detection | KS Test (scipy) | Production data shift monitoring |
| UI | Streamlit (6 pages) | Interactive demo dashboard |

---

## 📊 Model Results

| Task | Dataset | Training Samples | F1-Score | Accuracy | GPU |
|------|---------|-----------------|----------|----------|-----|
| **Hate Speech** | Davidson (EN) + HASOC 2019 (HI/Hinglish) | 13,246 | **78.74%** | 78.71% | Tesla P100-PCIE-16GB |
| **Misinformation** | FakeNewsNet (EN) | ~32,000 | **98.93%** | 98.93% | Tesla P100-PCIE-16GB |

> Training performed on Kaggle — see `notebooks/kaggle_training.ipynb`

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT DATASETS                            │
│                                                                   │
│   Davidson (EN)        HASOC 2019 (HI/Hinglish)   FakeNewsNet   │
│   12,970 tweets        5,983 posts                 40,587 news   │
│   [HuggingFace]        [data/raw/*.tsv + DVC]      [HuggingFace] │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    KAGGLE GPU TRAINING                            │
│                                                                   │
│      MuRIL Fine-tuning · Tesla P100-16GB · 5 Epochs              │
│      AdamW lr=1e-5 · Dropout(0.3) · FP16 Mixed Precision        │
│      notebooks/kaggle_training.ipynb                             │
└─────────────────────────────────────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
         ┌─────────┐     ┌──────────┐      ┌──────────┐
         │   DVC   │     │  MLflow  │      │  AWS S3  │
         │  Data   │     │ Exp Track│      │ Artifacts│
         │Version  │     │ Metrics  │      │  Model   │
         └─────────┘     └──────────┘      └──────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│               MODULAR SOURCE CODE (VS Code)                      │
│                                                                   │
│  src/data/      src/models/    src/training/    src/explainability│
│  src/bias/      src/monitoring/                                   │
└─────────────────────────────────────────────────────────────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                  ▼
      ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
      │  Streamlit   │  │    Docker    │  │  GitHub Actions  │
      │   UI Demo    │  │  Container   │  │   CI/CD (3 jobs) │
      │  (6 pages)   │  │  AWS ECR     │  │  Test→Build→Deploy│
      └──────────────┘  └──────────────┘  └──────────────────┘
                                                    │
                                                    ▼
                                         ┌──────────────────┐
                                         │    AWS EC2       │
                                         │  t2.micro        │
                                         │  ap-south-1      │
                                         │  :8501 Live      │
                                         └──────────────────┘
```

---

## 📁 Project Structure

```
multilingual-hate-misinfo/
│
├── 📓 notebooks/
│   └── kaggle_training.ipynb        ← Complete GPU training notebook (run on Kaggle)
│
├── 🖥️ app/
│   └── streamlit_app.py             ← 6-page Streamlit dashboard
│
├── ⚙️ configs/
│   └── config.yaml                  ← All hyperparameters, paths, MLflow settings
│
├── 🧠 src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                ← Dataset loaders (Davidson, HASOC, FakeNewsNet)
│   │   └── preprocessor.py         ← MuRIL tokenizer, PyTorch Dataset
│   │
│   ├── models/
│   │   ├── muril_classifier.py      ← MuRIL + Dropout + Linear head
│   │   ├── evaluator.py             ← Metrics, confusion matrix, reports
│   │   └── batch_predictor.py      ← Batch deployment strategy
│   │
│   ├── training/
│   │   └── trainer.py              ← Training loop + MLflow logging
│   │
│   ├── explainability/
│   │   └── lime_explainer.py       ← LIME word importance
│   │
│   ├── bias/
│   │   └── bias_analyzer.py        ← Equalized odds across language groups
│   │
│   └── monitoring/
│       └── drift_detector.py       ← KS-test drift detection
│
├── 🧪 tests/
│   ├── test_data.py                 ← 10 unit tests (data loading, config)
│   └── test_model.py               ← 9 unit tests (metrics, bias, drift)
│
├── 📦 models/
│   └── saved/
│       ├── muril_hate_best.pt      ← Trained hate speech model (953MB — not in Git)
│       └── muril_misinfo_best.pt   ← Trained misinfo model (953MB — not in Git)
│
├── 📊 outputs/
│   ├── confusion_matrix_hate.png
│   ├── confusion_matrix_misinfo.png
│   ├── sample_predictions_hate.csv
│   └── sample_predictions_misinfo.csv
│
├── 📋 data/
│   └── raw/
│       ├── hasoc_train.tsv         ← DVC tracked (not in Git directly)
│       ├── hasoc_train.tsv.dvc     ← DVC pointer file (committed to Git)
│       ├── hasoc_test.tsv          ← DVC tracked
│       └── hasoc_test.tsv.dvc      ← DVC pointer file
│
├── 🔄 .github/
│   └── workflows/
│       └── ci_cd.yml               ← 3-stage GitHub Actions pipeline
│
├── 🐳 Dockerfile                    ← python:3.11-slim based image
├── 🚫 .dockerignore                 ← Excludes models, mlruns, venv
├── 🔁 dvc.yaml                     ← 3-stage DVC pipeline definition
├── 📌 .dvc/config                  ← DVC remote storage config
├── 📜 requirements.txt             ← All Python dependencies
└── 📖 README.md                    ← This file
```

---

## 📂 Datasets

| Dataset | Language | Task | Samples | Source |
|---------|----------|------|---------|--------|
| Davidson Hate Speech | English | Hate Speech | 12,970 | HuggingFace `tweet_eval/hate` |
| HASOC 2019 | Hindi / Hinglish | Hate Speech | 5,983 | [HASOC Fire](https://hasocfire.github.io/) (manual download required) |
| FakeNewsNet | English | Misinformation | 40,587 | HuggingFace `GonzaloA/fake_news` |

> **Note:** Davidson and FakeNewsNet download automatically via HuggingFace.
> HASOC requires manual download — see [Step 4](#step-4--setup-hasoc-dataset-dvc) below.

---

## 🚀 Quick Start — Full Setup Guide

Follow these steps **in order**. Each step depends on the previous one.

---

### Step 1 — Prerequisites

Make sure you have these installed:

```bash
# Check Python version (need 3.11)
python --version

# Check Git
git --version

# Check pip
pip --version
```

**Required software:**
- Python 3.11+ → https://www.python.org/downloads/
- Git → https://git-scm.com/
- Docker Desktop (for Docker steps) → https://www.docker.com/products/docker-desktop/
- VS Code (recommended) → https://code.visualstudio.com/

---

### Step 2 — Clone & Virtual Environment

```bash
# 1. Clone the repository
git clone https://github.com/Jay-Modhiya/multilingual-hate-misinfo.git
cd multilingual-hate-misinfo

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate

# You should see (venv) prefix in terminal
```

---

### Step 3 — Install Dependencies

```bash
# Install all required packages
pip install --upgrade pip
pip install -r requirements.txt
```

> ⚠️ **This installs:** PyTorch (CPU), transformers, datasets, streamlit, mlflow, dvc, lime, scikit-learn, scipy, matplotlib, seaborn, pytest, and more.
>
> ⏱️ **Time:** ~3-5 minutes depending on internet speed.

---

### Step 4 — Setup HASOC Dataset (DVC)

The HASOC Hindi dataset requires **manual download** (data agreement required).

**Option A — Manual Download:**

1. Go to: https://hasocfire.github.io/hasoc/2019/dataset.html
2. Download `hasoc_train.tsv` and `hasoc_test.tsv`
3. Place both files in `data/raw/`:

```bash
# Create directory if it doesn't exist
mkdir -p data/raw

# Place the files here:
# data/raw/hasoc_train.tsv
# data/raw/hasoc_test.tsv
```

**Option B — DVC Pull (if you have DVC remote access):**

```bash
# Initialize DVC (already done — just pull data)
dvc pull

# Verify files exist
python -c "
import pandas as pd
df = pd.read_csv('data/raw/hasoc_train.tsv', sep='\t')
print(f'HASOC train loaded: {len(df)} rows')
"
```

**Verify datasets are ready:**

```bash
python -c "
from src.data.loader import load_config, load_all_datasets
cfg = load_config('configs/config.yaml')
datasets = load_all_datasets(cfg)
for name, splits in datasets.items():
    print(f'{name}: train={len(splits[\"train\"])} val={len(splits[\"val\"])} test={len(splits[\"test\"])}')
"
```

Expected output:
```
davidson: train=9079  val=1297  test=2594
hasoc:    train=4187  val=599   test=1197
fakenews: train=...   val=...   test=...
```

---

### Step 5 — Run the Streamlit App

> ⚠️ You need the trained model files in `models/saved/` for full functionality.
> Download them from the Kaggle training notebook output or contact the authors.

```bash
# Run the Streamlit dashboard
python -m streamlit run app/streamlit_app.py
```

Open your browser at: **http://localhost:8501**

**What you will see:**
- 🏠 **Home** — Project overview, datasets, MLOps stack
- 🔍 **Detect** — Live hate speech + misinformation detection (English, Hindi, Hinglish)
- 📊 **Model Results** — Confusion matrices, performance metrics, training curves
- ⚖️ **Bias Analysis** — Equalized odds across English vs Hindi/Hinglish
- 📡 **Monitoring** — KS-test data drift detection with history
- 🔢 **Batch Predict** — Batch inference with CSV download

---

### Step 6 — View MLflow Experiments

MLflow tracks all training runs, metrics, and artifacts.

```bash
# Start MLflow UI
python -m mlflow ui --port 5000
```

Open your browser at: **http://localhost:5000**

**What to look at in MLflow:**

```
Experiments
├── muril-multilingual-hate-hate          ← Hate speech experiment
│   └── run: muril-hate-20260414-2027
│       ├── Parameters: lr, epochs, batch_size, model...
│       ├── Metrics: val_f1, val_accuracy, train_loss (per epoch)
│       └── Artifacts:
│           ├── muril_hate_best.pt        ← Model weights
│           ├── confusion_matrix_hate.png ← Visual result
│           ├── sample_predictions_hate.csv
│           └── config.yaml
│
└── muril-multilingual-hate-misinfo       ← Misinformation experiment
    └── run: muril-misinfo-20260414-2049
        ├── Metrics: val_f1=98.93%, val_accuracy=98.93%
        └── Artifacts: model, confusion matrix, predictions
```

---

### Step 7 — Run DVC Pipeline

DVC manages the reproducible training pipeline.

```bash
# See pipeline stages (dry run — shows what would run)
dvc repro --dry

# Run the full pipeline (trains both models)
# ⚠️ WARNING: This runs on CPU and will be VERY slow (~hours)
# Use Kaggle notebook for actual training instead
dvc repro

# Check pipeline status
dvc status

# Show pipeline DAG
dvc dag
```

**Pipeline stages:**

```
prepare → train_hate → [outputs: muril_hate_best.pt]
prepare → train_misinfo → [outputs: muril_misinfo_best.pt]
```

**DVC tracked files:**

```bash
# See what DVC is tracking
dvc list . --dvc-only

# Check data file status
dvc status data/raw/hasoc_train.tsv.dvc
```

---

### Step 8 — Run Unit Tests

The project has **19 unit tests** covering all major components.

```bash
# Run all tests
pytest tests/ -v

# Run only data tests
pytest tests/test_data.py -v

# Run only model tests
pytest tests/test_model.py -v

# Run with coverage report
pytest tests/ -v --tb=short
```

**Expected output:**
```
tests/test_data.py::test_config_loads                    PASSED
tests/test_data.py::test_config_values                   PASSED
tests/test_data.py::test_cleaner_english                 PASSED
tests/test_data.py::test_cleaner_hindi                   PASSED
tests/test_data.py::test_hasoc_loads                     PASSED
tests/test_data.py::test_fakenewsnet_loads               PASSED
tests/test_data.py::test_label_distribution              PASSED
tests/test_data.py::test_train_val_test_no_overlap       SKIPPED (placeholder data)
tests/test_data.py::test_config_aws_region               PASSED
tests/test_data.py::test_config_model_name               PASSED
tests/test_model.py::test_compute_metrics                PASSED
tests/test_model.py::test_perfect_predictions            PASSED
tests/test_model.py::test_build_summary_table            PASSED
tests/test_model.py::test_group_metrics                  PASSED
tests/test_model.py::test_equalized_odds                 PASSED
tests/test_model.py::test_extract_features               PASSED
tests/test_model.py::test_drift_no_drift                 PASSED
tests/test_model.py::test_drift_with_drift               PASSED
tests/test_model.py::test_drift_log                      PASSED

=================== 18 passed, 1 skipped in 15.11s ===================
```

---

### Step 9 — Build Docker Image

```bash
# Make sure Docker Desktop is running first!

# Build the Docker image
docker build -t muril-hate-misinfo .

# Verify image was created
docker images | grep muril-hate-misinfo

# Run locally with Docker
docker run -d \
  --name muril-app \
  -p 8501:8501 \
  muril-hate-misinfo

# Check container is running
docker ps

# Open app at:  http://localhost:8501

# Stop the container
docker stop muril-app
docker rm muril-app
```

---

## 🧰 MLOps Stack

### DVC — Data Version Control

```bash
# Initialize DVC (already done in this repo)
dvc init

# Track a data file
dvc add data/raw/hasoc_train.tsv

# Check what's tracked
dvc list . --dvc-only

# Reproduce pipeline
dvc repro

# Push data to remote
dvc push

# Pull data from remote
dvc pull
```

### MLflow — Experiment Tracking

```bash
# Start UI
python -m mlflow ui --port 5000

# Set experiment name in code:
mlflow.set_experiment("my-experiment")

# Log a run:
with mlflow.start_run():
    mlflow.log_param("lr", 1e-5)
    mlflow.log_metric("val_f1", 78.74)
    mlflow.log_artifact("outputs/confusion_matrix.png")
```

### GitHub Actions — CI/CD

The pipeline runs automatically on every `git push` to `main`:

```
Push to main
    │
    ▼
Job 1: Run Tests (59s)
    │  ✅ pytest tests/ -v
    │
    ▼
Job 2: Build & Push to AWS ECR (11 min)
    │  ✅ docker build --no-cache
    │  ✅ docker push ECR:latest
    │
    ▼
Job 3: Deploy to AWS EC2 (2 min)
    ✅ SSH → docker pull → docker run
```

**Required GitHub Secrets:**

| Secret | Value |
|--------|-------|
| `AWS_ACCESS_KEY_ID` | Your AWS IAM Access Key |
| `AWS_SECRET_ACCESS_KEY` | Your AWS IAM Secret Key |
| `EC2_HOST` | EC2 Public IP (changes on restart!) |
| `EC2_SSH_KEY` | Contents of `.pem` key file |

---

## 🖥️ Streamlit App Pages

| Page | Icon | What it shows |
|------|------|---------------|
| Home | 🏠 | Project overview, datasets, MLOps stack cards |
| Detect | 🔍 | Live prediction — type English/Hindi/Hinglish text, click sample buttons |
| Model Results | 📊 | Confusion matrices, F1 per epoch, performance table |
| Bias Analysis | ⚖️ | English vs Hindi/Hinglish F1, FPR, FNR gaps |
| Monitoring | 📡 | Paste production texts → KS-test drift detection |
| Batch Predict | 🔢 | Multi-text prediction table + CSV download |

**Sample texts to try in Detect page:**

```
# English hate (detected ~87-91% confidence)
"You people don't belong here, go back to your country."

# Hindi hate (detected ~86-89% confidence)  
"तुम जैसे लोगों को यहाँ नहीं रहना चाहिए।"

# Hinglish hate (borderline ~51-55%)
"Yeh log bahut bure hain, inhe bahar nikalo."

# Fake news (check misinformation)
"Scientists confirm 5G towers are spreading the virus to control population."
```

---

## 🔢 Batch Deployment Strategy

**Why Batch over Real-time or Streaming?**

| Strategy | Verdict | Reason |
|----------|---------|--------|
| ✅ **Batch** (chosen) | Best fit | Content moderation processes thousands of posts at scheduled intervals. Efficient GPU use, lower cost, simpler infrastructure. |
| ❌ Real-time | Not needed | Would require always-on GPU ($$$). Moderation doesn't need millisecond response. |
| ❌ Streaming | Overkill | Kafka/Kinesis complexity not justified. Batch runs are sufficient. |

**Run batch prediction demo:**

```bash
python -m src.models.batch_predictor
```

**Sample output:**
```
BATCH PREDICTION DEMO
Deployment Strategy: BATCH
Model: google/muril-base-cased

── HATE SPEECH DETECTION BATCH ──
Text                                    Prediction   Confidence
You people don't belong here...         hate         87.4%
I love watching cricket...              not_hate     91.1%
तुम जैसे लोगों को यहाँ नहीं...         hate         88.5%
आज मौसम बहुत अच्छा है।                not_hate     90.4%

Summary: 3/10 flagged as hate (30.0%)
Results saved → outputs/batch_hate_results.csv
```

---

## ⚖️ Bias & Fairness Analysis

The model shows a **performance gap** between language groups — this is documented and expected:

| Language Group | Samples | F1-Score | FPR | FNR |
|---------------|---------|----------|-----|-----|
| English | 2,594 | 74.61% | 20.64% | 31.78% |
| Hindi / Hinglish | 1,197 | 83.36% | 11.86% | 21.14% |

**Equalized Odds Gaps:**
- F1 Gap: **8.75%**
- FPR Gap: **8.78%**
- FNR Gap: **10.64%**

> **Academic context:** The gap is expected — Davidson (EN) is 2.2× larger than HASOC (HI). Bias documentation is a responsible AI practice, not a failure.

---

## 📡 Data Drift Monitoring

**How the KS-test drift detector works:**

```
Training data reference distribution
         │
         │  Fit on 2000 Davidson training texts
         ▼
Features extracted per text:
  - text_length    (character count)
  - word_count     (token count)
  - avg_word_length
  - upper_ratio    (fraction uppercase)
  - punct_ratio    (fraction punctuation)
         │
         │  KS test (p < 0.05 → drift)
         ▼
Production incoming texts
```

**Test drift detection yourself:**

```bash
# Paste this in the Streamlit Monitoring page to TRIGGER drift:
ALERT!!! URGENT MESSAGE TO ALL CITIZENS!!!
BREAKING NEWS: EVERYTHING IS FAKE!!!
WARNING!!! DO NOT TRUST THE GOVERNMENT!!!

# Paste this to show NO drift:
The weather is nice today in Mumbai.
I love watching cricket with friends.
Government announces new infrastructure budget.
```

---

## 🔄 CI/CD Pipeline

Full pipeline defined in `.github/workflows/ci_cd.yml`:

```yaml
# Triggered on every push to main branch
on:
  push:
    branches: [ main ]

jobs:
  test:      # Job 1: Run 19 pytest tests
  build:     # Job 2: docker build + push to AWS ECR
  deploy:    # Job 3: SSH into EC2, pull image, restart container
```

**View pipeline status:**
→ GitHub Repository → **Actions** tab → Latest run

**Pipeline #17 result (latest passing):**
```
✅ Run Tests         — 59s   — 18 passed, 1 skipped
✅ Build & Push ECR  — 11min — Image pushed to 679409142377.dkr.ecr.ap-south-1.amazonaws.com
✅ Deploy to EC2     — 2min  — Container running on port 8501
```

---

## ☁️ AWS Deployment

| Component | Value |
|-----------|-------|
| EC2 Instance | `i-0de255b75fced01be` (muril-hate-misinfo-app) |
| Instance Type | t2.micro — 1 vCPU, 1 GB RAM (Free Tier) |
| Region | Asia Pacific Mumbai — `ap-south-1` |
| ECR Registry | `679409142377.dkr.ecr.ap-south-1.amazonaws.com/muril-hate-misinfo` |
| App Port | 8501 |
| Volume | 20 GiB gp3 EBS |

**⚠️ Important:** EC2 Public IP **changes every time** you stop/start the instance.
After restarting EC2, update the `EC2_HOST` secret in GitHub → Settings → Secrets.

**Access the live app:**
```
http://[EC2_PUBLIC_IP]:8501
```

**Start/Stop EC2 (to save credits):**
```
AWS Console → EC2 → Instances → Select instance → Instance State → Start/Stop
```

---

## 🎓 Viva — Key Talking Points

**Why MuRIL over DistilBERT (CA-3)?**
> *"MuRIL = Multilingual Representations for Indian Languages, pre-trained by Google on 17 Indian languages including transliterated Hinglish. Unlike DistilBERT which is English-only, MuRIL natively understands Hindi script and code-switched text — essential for Indian social media moderation."*

**Why F1=78.74% (lower than CA-3's 92%)?**
> *"CA-3 DistilBERT worked on English-only data. CA-4 MuRIL works across 3 languages simultaneously. Multilingual classification on mixed-script data is inherently harder. 78.74% is consistent with state-of-the-art multilingual hate speech benchmarks."*

**What does DVC do?**
> *"DVC tracks the HASOC dataset files (5,983 rows) the same way Git tracks code — with a checksum pointer. If the data changes, dvc status shows it. dvc repro reruns only the affected pipeline stages. This ensures reproducibility."*

**What does MLflow track?**
> *"MLflow logs every training run: hyperparameters (lr, batch_size, epochs), per-epoch validation metrics (F1, accuracy, loss), and artifacts (model .pt file, confusion matrix PNG, sample predictions CSV, config YAML). You can compare runs and reproduce any experiment."*

**Why Batch over Real-time deployment?**
> *"Content moderation platforms process thousands of posts in scheduled batches — not one at a time. Batch inference is 10x more GPU-efficient than individual requests. Real-time would require always-on GPU infrastructure ($$$) which isn't justified for this use case."*

**What is equalized odds in your bias analysis?**
> *"Equalized odds requires that a model's False Positive Rate and False Negative Rate be equal across demographic groups. Our model shows an 8.78% FPR gap between English and Hindi/Hinglish — higher FPR on English because Davidson dataset has 77% offensive class dominance vs. HASOC's balanced distribution. We document this transparently."*

---

## 📋 Project Report

The full CA-4 MLOps project report is available in the repository:

**File:** `25070149031_Jay_Modhiya_MLOPs_CA-4_Mini_Project_report.docx`

---

## 👥 Authors

| Name | Roll Number | Email |
|------|------------|-------|
| **Jay Modhiya** | 25070149031 | jaymodhiya2@gmail.com |
| **Krishna Nandi** | 25070149011 | — |

**Institute:** Symbiosis Institute of Technology, Pune
**Programme:** MTech Artificial Intelligence & Machine Learning
**Subject:** NLPA + MLOps (CA-4 Mini Project)
**Academic Year:** 2025–2026
**Supervisor:** Dr. Anupkumar M Bongale

---

## 📜 License

This project is developed for academic purposes at Symbiosis Institute of Technology, Pune.

---

<div align="center">

**⭐ If this project helped you, please give it a star!**

*Built with ❤️ using MuRIL, MLflow, DVC, Docker, GitHub Actions, and AWS*

</div>
