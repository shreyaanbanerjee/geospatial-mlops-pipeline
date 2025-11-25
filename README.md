# Planet Sentinel: Geospatial MLOps Pipeline

**Planet Sentinel** is a prototype automated MLOps pipeline for deforestation detection using Sentinel-2 satellite imagery. It demonstrates an end-to-end workflow from data ingestion to model deployment and monitoring.

## Features
*   **Pillar 1 (Data)**: Automated ingestion from Google Earth Engine, labeling support, and DVC versioning.
*   **Pillar 2 (Training)**: Lightweight Siamese U-Net model, MLflow tracking, and automated training on Kaggle.
*   **Pillar 3 (Deployment)**: FastAPI service, Docker containerization, and automated deployment via GitHub Actions.
*   **Pillar 4 (Monitoring)**: Data drift detection triggering automated retraining.

## Prerequisites
1.  **Python 3.9+** installed.
2.  **Docker** installed and running.
3.  **Google Earth Engine (GEE)** account and project.
4.  **Kaggle** account (for training).
5.  **GitHub** repository with Actions enabled.

## Setup
Install the required Python packages:
```bash
pip install -r requirements.txt
pip install dvc mlflow kaggle rasterio
```

## Usage Guide

### 1. Data Pipeline (Bootstrap)
**Authenticate GEE:**
```bash
earthengine authenticate
```

**Ingest Data:**
Edit `ingest/gee_ingest.py` to set your GCP Project ID, then run:
```bash
python ingest/gee_ingest.py --project YOUR_PROJECT_ID --aoi aoi/india_wayanad.geojson --before 2024-01-01 2024-01-31 --after 2024-11-01 2024-11-30 --name india_wayanad --drive-folder EO_Exports
```
*Note: Download exported images from Drive to `data/raw`.*

**Preprocess (Chip):**
```bash
python preprocess/chip_dataset.py --before data/raw/india_wayanad_before.tif --after data/raw/india_wayanad_after.tif --mask data/raw/india_wayanad_mask.tif --out-dir data/chips
```

**Create Baseline:**
```bash
python monitor/create_baseline.py --chips-dir data/chips --out baseline.json
```

### 2. Training (Automated)
Run locally:
```bash
python train/train.py --data-dir data/chips --epochs 5
```
Or trigger via **GitHub Actions** -> **Train Model**.

### 3. Deployment (Automated)
The `deploy.yaml` workflow runs on a schedule. To test locally:
1.  Start a self-hosted runner.
2.  Wait for the workflow to trigger (or trigger manually).
3.  Test the API:
    ```bash
    curl -X POST "http://localhost:8000/predict" -F "file=@data/chips/tile_00000_before.tif"
    ```

# Planet Sentinel: Geospatial MLOps Pipeline

**Planet Sentinel** is a prototype automated MLOps pipeline for deforestation detection using Sentinel-2 satellite imagery. It demonstrates an end-to-end workflow from data ingestion to model deployment and monitoring.

## Features
*   **Pillar 1 (Data)**: Automated ingestion from Google Earth Engine, labeling support, and DVC versioning.
*   **Pillar 2 (Training)**: Lightweight Siamese U-Net model, MLflow tracking, and automated training on Kaggle.
*   **Pillar 3 (Deployment)**: FastAPI service, Docker containerization, and automated deployment via GitHub Actions.
*   **Pillar 4 (Monitoring)**: Data drift detection triggering automated retraining.

## Prerequisites
1.  **Python 3.9+** installed.
2.  **Docker** installed and running.
3.  **Google Earth Engine (GEE)** account and project.
4.  **Kaggle** account (for training).
5.  **GitHub** repository with Actions enabled.

## Setup
Install the required Python packages:
```bash
pip install -r requirements.txt
pip install dvc mlflow kaggle rasterio
```

## Usage Guide

### 1. Data Pipeline (Bootstrap)
**Authenticate GEE:**
```bash
earthengine authenticate
```

**Ingest Data:**
Edit `ingest/gee_ingest.py` to set your GCP Project ID, then run:
```bash
python ingest/gee_ingest.py --project YOUR_PROJECT_ID --aoi aoi/india_wayanad.geojson --before 2024-01-01 2024-01-31 --after 2024-11-01 2024-11-30 --name india_wayanad --drive-folder EO_Exports
```
*Note: Download exported images from Drive to `data/raw`.*

**Preprocess (Chip):**
```bash
python preprocess/chip_dataset.py --before data/raw/india_wayanad_before.tif --after data/raw/india_wayanad_after.tif --mask data/raw/india_wayanad_mask.tif --out-dir data/chips
```

**Create Baseline:**
```bash
python monitor/create_baseline.py --chips-dir data/chips --out baseline.json
```

### 2. Training (Automated)
Run locally:
```bash
python train/train.py --data-dir data/chips --epochs 5
```
Or trigger via **GitHub Actions** -> **Train Model**.

### 3. Deployment (Automated)
The `deploy.yaml` workflow runs on a schedule. To test locally:
1.  Start a self-hosted runner.
2.  Wait for the workflow to trigger (or trigger manually).
3.  Test the API:
    ```bash
    curl -X POST "http://localhost:8000/predict" -F "file=@data/chips/tile_00000_before.tif"
    ```

### 4. Monitoring (Simulation)
Simulate drift to trigger retraining:
```bash
python monitor/monitor.py --baseline baseline.json --new-data data/new_chips --repo YOUR_GITHUB_USER/REPO --token YOUR_GITHUB_TOKEN
```

## Documentation
*   [Requirements (FAQ)](docs/PLANET_SENTINEL_FAQ.md)
*   [Labeling Guide](docs/labeling_guide.md)
*   [Implementation Plan](docs/implementation_plan.md)
*   [Walkthrough](docs/walkthrough.md)