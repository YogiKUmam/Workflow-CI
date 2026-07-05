# Workflow-CI

Repository CI SMSML untuk retraining otomatis model Heart Disease Classification memakai MLflow Project dan GitHub Actions.

- `MLProject/` - MLflow Project (script training, MLproject, conda.yaml, dataset preprocessed).
- `.github/workflows/retrain.yml` - workflow retraining: run MLflow Project, upload artefak, build & push Docker image ke Docker Hub.

Secrets yang dibutuhkan:

- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
