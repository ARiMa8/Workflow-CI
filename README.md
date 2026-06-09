# Workflow-CI - Credit Scoring MLflow Project

Repository ini digunakan untuk Kriteria 3 submission Dicoding kelas Membangun Sistem Machine Learning.

## Isi Repository

- `.github/workflows/mlflow-ci.yml`: workflow CI untuk retraining model, upload artifact, build Docker image, dan push ke Docker Hub.
- `MLProject/MLProject`: definisi MLflow Project.
- `MLProject/conda.yaml`: environment MLflow Project.
- `MLProject/modelling.py`: script training otomatis memakai dataset hasil preprocessing.
- `MLProject/credit_scoring_preprocessing`: dataset siap latih dari Kriteria 1 dan 2.
- `MLProject/Tautan ke Docker Hub.txt`: link Docker Hub image.

## Secrets GitHub Actions

Tambahkan secrets berikut pada repository GitHub:

```text
DOCKERHUB_USERNAME=arima88
DOCKERHUB_TOKEN=<token Docker Hub>
```

## Menjalankan Lokal

```bash
cd Workflow-CI
pip install -r MLProject/requirements.txt
mlflow run MLProject --env-manager=local
```

## Docker Image

Image yang dibuat workflow:

```text
arima88/credit-scoring-msml:latest
```
