# MLProject Heart Disease CI

MLflow Project untuk retraining otomatis model Heart Disease Classification.

Run lokal:

```bash
mlflow run . --env-manager=local
```

Output model lokal disimpan di `outputs/model`. Workflow GitHub Actions `retrain.yml` menjalankan project ini, mengunggah artefak training, lalu build dan push Docker image ke Docker Hub.
