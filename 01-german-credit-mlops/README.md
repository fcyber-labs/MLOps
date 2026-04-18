# German Credit Risk MLOps Pipeline

## Project Overview

This project implements a complete end-to-end MLOps pipeline for credit risk classification using the German Credit Dataset. The pipeline demonstrates industry best practices including data versioning, experiment tracking, hyperparameter optimization, model evaluation, and explainable AI.

## Key Features

- **Data Version Control (DVC)**: Track and version datasets, models, and evaluation outputs
- **Experiment Tracking (MLflow)**: Log parameters, metrics, and artifacts on Databricks Community Edition
- **Hyperparameter Tuning**: GridSearchCV with 120 combinations (5 depths × 6 estimators × 4 features)
- **Model Performance**: Random Forest achieving 73% ROC-AUC and 73% accuracy
- **Model Explainability**: SHAP (SHapley Additive exPlanations) for interpretability
- **Reproducible Pipeline**: Fully automated workflow with DVC

## Pipeline Stages

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| **Preprocess** | `preprocess.py` | `german_credit_data_risk.csv` | `features.csv` |
| **Train** | `train.py` | `features.csv` | `model.pkl`, `X_test.csv`, `y_test.csv` |
| **Evaluate** | `evaluate.py` | `model.pkl`, `X_test.csv`, `y_test.csv` | `metrics.json`, `roc_curve.png` |
| **Explain** | `explain.py` | `X_test.csv` | `shap_summary.png` |

## Model Performance

### Best Hyperparameters Found

| Parameter | Optimal Value |
|-----------|---------------|
| max_depth | 10 |
| max_features | 4 |
| n_estimators | 150 |

### Model Metrics

| Metric | Cross-Validation | Test Set |
|--------|------------------|----------|
| ROC-AUC | 0.7479 | 0.7344 |
| Accuracy | - | 0.7320 |

### Performance Interpretation

- **ROC-AUC 0.7344**: The model is significantly better than random (0.5) and shows good discriminatory power for credit risk classification
- **Accuracy 73.2%**: The model correctly classifies approximately 3 out of 4 credit applicants

## Technology Stack

| Tool | Purpose |
|------|---------|
| **Python 3.12** | Programming language |
| **DVC** | Data and pipeline versioning |
| **MLflow** | Experiment tracking (Databricks) |
| **Scikit-learn** | Model training & evaluation |
| **SHAP** | Model explainability |
| **Matplotlib** | Visualizations |
| **Git/GitHub** | Code version control |

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/fcyber-labs/MLOps.git
cd MLOps/01-german-credit-mlops
```

### 2. Create virtual environment

**Using venv (macOS/Linux):**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Using venv (Windows):**

```bash
python -m venv venv
venv\Scripts\activate
```

**Using conda:**

```bash
conda create -n credit-risk python=3.12
conda activate credit-risk
```

**Using uv (fast alternative):**

```bash
uv venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or with uv:

```bash
uv pip install -r requirements.txt
```

### 4. Set up environment variables for Databricks MLflow

Create a `.env` file in the project root:

```bash
DATABRICKS_USERNAME=your_email@gmail.com
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
DATABRICKS_TOKEN=your_personal_access_token
```

### 5. Initialize DVC (if not already initialized)

```bash
dvc init
```

### 6. Configure DVC remote (optional, for pushing data)

```bash
dvc remote add origin https://dagshub.com/fcyber/demo.dvc
dvc remote default origin
```

## Usage

### Run the complete pipeline

```bash
dvc repro
```

### Run individual stages

```bash
dvc repro preprocess
dvc repro train
dvc repro evaluate
dvc repro explain
```

### View MLflow experiments on Databricks

1. Visit your Databricks workspace
2. Click on "Machine Learning" → "Experiments"
3. Select the "german-credit" experiment
4. View all runs, parameters, metrics, and artifacts

### Run MLflow locally (alternative)

```bash
mlflow ui --backend-store-uri file:./mlruns
```

Then open http://localhost:5000

### Generate pipeline DAG visualization

```bash
dvc dag
```

## Output Artifacts

### Logged to MLflow (Databricks)

- `confusion_matrix.txt` - Confusion matrix of test predictions
- `classification_report.txt` - Precision, recall, F1-score by class
- `grid_search_results.csv` - All 120 hyperparameter combinations with scores
- `feature_importance.png` - Top 15 feature importances visualization
- `params_summary.json` - Complete training configuration summary
- `best_model` - Registered Random Forest model

### Tracked by DVC (Local)

- `models/model.pkl` - Trained Random Forest model
- `reports/metrics.json` - Evaluation metrics in JSON format
- `reports/roc_curve.png` - ROC curve with AUC score
- `reports/shap_summary.png` - SHAP feature importance summary plot
- `data/processed/X_test.csv` - Test features (25% holdout)
- `data/processed/y_test.csv` - Test labels

## Pipeline Visualization

The pipeline structure as shown by `dvc dag`:

```text
        +-------------+
        | preprocess  |
        +-------------+
              |
              | (features.csv)
              v
        +-------------+
        |    train    |
        +-------------+
           /        \
          /          \
         | (model)    | (X_test)
         |            |
         v            v
  +---------+    +---------+
  | evaluate|    | explain |
  +---------+    +---------+
         |            |
         | (metrics)  | (SHAP plot)
         v            v
    +---------+  +---------+
    | reports |  | reports |
    +---------+  +---------+
```

## Model Interpretation (SHAP Analysis)

Based on the trained Random Forest model, the top 5 most influential features for credit risk prediction are:

1. **Credit amount** - Higher amounts indicate higher risk
2. **Duration** - Longer loan terms increase risk
3. **Age** - Younger applicants typically show higher risk
4. **Purpose of loan** - Some purposes have higher default rates
5. **Savings account balance** - Higher savings indicate lower risk

The SHAP summary plot in `reports/shap_summary.png` shows the direction and magnitude of each feature's impact on predictions.

## Troubleshooting

### DVC remote authentication issues

```bash
dvc remote modify origin --local access_key_id your_username
dvc remote modify origin --local secret_access_key your_token
```

### MLflow connection issues

Verify environment variables:

```bash
source .env
echo $DATABRICKS_HOST
echo $DATABRICKS_TOKEN
```

Test connection:

```bash
python -c "import mlflow; mlflow.set_tracking_uri('databricks'); print(mlflow.get_tracking_uri())"
```

### Pipeline reproduction issues

Force reprocess all stages:

```bash
dvc repro --force
```

Check pipeline status:

```bash
dvc status
```

Clean DVC cache if needed:

```bash
dvc gc --workspace --force
```

### Module not found errors

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Reinstall specific package:

```bash
pip install --upgrade scikit-learn pandas numpy
```

## Future Improvements

- Deploy model as REST API using FastAPI
- Add CI/CD pipeline with GitHub Actions
- Implement model monitoring with Evidently AI
- Add A/B testing framework
- Automate retraining on new data
- Add Docker containerization
- Implement feature store integration
- Add model versioning in MLflow registry

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Author

**fcyber** - GitHub Profile

## Acknowledgments

- German Credit Dataset from UCI Machine Learning Repository
- DVC for pipeline orchestration
- MLflow for experiment tracking
- Databricks for free Community Edition hosting
- SHAP library for model interpretability

## Repository Links

- **GitHub**: https://github.com/fcyber-labs/MLOps
- **MLflow Tracking**: Hosted on Databricks Community Edition
- **Code Version**: Python 3.12+

---

*This project demonstrates a production-ready MLOps pipeline with reproducibility, experiment tracking, hyperparameter optimization, and model explainability best practices.*