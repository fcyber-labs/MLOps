import pandas as pd
import numpy as np
import json
import mlflow
import mlflow.sklearn
import joblib
from mlflow.tracking import MlflowClient
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve
import matplotlib.pyplot as plt
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set MLflow tracking from environment variables
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI", "")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", ""))


print("Starting evaluation...")


# Load test data
print("\n Loading test data...")
X_test = pd.read_csv("data/processed/X_test.csv")
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()
print(f" Test data loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")

# Try to load model - first from local file, then from MLflow
best_model = None
model_source = None

# Option 1: Load from local file (preferred for DVC pipeline)
print("\n Attempting to load model from local file...")
local_model_path = "models/model.pkl"
if os.path.exists(local_model_path):
    try:
        best_model = joblib.load(local_model_path)
        model_source = "local file"
        print(f" Model loaded from {local_model_path}")
    except Exception as e:
        print(f" Could not load model from local file: {e}")
else:
    print(f" Local model file not found at {local_model_path}")

# Option 2: Fall back to MLflow if local model not available
if best_model is None:
    print("\n🔍 Attempting to load best model from MLflow...")
    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name("Default")
        
        # Handle case where experiment might not exist
        if experiment is None:
            print(" Experiment 'Default' not found. Creating it...")
            experiment = client.create_experiment("Default")
        
        runs = client.search_runs(experiment.experiment_id)
        
        if len(runs) == 0:
            print(" No runs found in experiment. Please run training first.")
            sys.exit(1)
        
        best_run = sorted(
            runs,
            key=lambda x: x.data.metrics.get("roc_auc", 0),
            reverse=True
        )[0]
        
        best_run_id = best_run.info.run_id
        print(f" Best run ID: {best_run_id}")
        print(f"   Best run ROC-AUC: {best_run.data.metrics.get('roc_auc', 0):.4f}")
        
        model_uri = f"runs:/{best_run_id}/model"
        best_model = mlflow.sklearn.load_model(model_uri)
        model_source = "MLflow"
        print(f" Model loaded from MLflow (run: {best_run_id})")
    except Exception as e:
        print(f" Failed to load model from MLflow: {e}")
        print("   Please run training first: dvc repro train")
        sys.exit(1)

# Make predictions
print("\n Making predictions...")
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]

# Calculate metrics
print(" Calculating metrics...")
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_pred_proba)
}

# Display results
print("EVALUATION RESULTS")

for metric, value in metrics.items():
    print(f"  {metric.upper():12s}: {value:.4f}")


# Save metrics to JSON
print("\n Saving metrics...")
os.makedirs("reports", exist_ok=True)
with open("reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print(" Metrics saved to reports/metrics.json")

# Plot ROC curve
print(" Generating ROC curve...")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {metrics["roc_auc"]:.3f})', linewidth=2)
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve - Credit Risk Model', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig("reports/roc_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print(" ROC curve saved to reports/roc_curve.png")

# Log metrics to MLflow (only if we have credentials and model came from there)
print("\n Logging to MLflow...")
try:
    if os.getenv("MLFLOW_TRACKING_PASSWORD"):
        with mlflow.start_run(run_name="evaluation", nested=True):
            mlflow.log_metrics(metrics)
            mlflow.log_artifact("reports/metrics.json")
            mlflow.log_artifact("reports/roc_curve.png")
            mlflow.log_param("model_source", model_source)
            print(" Evaluation metrics logged to MLflow")
    else:
        print(" No MLflow credentials found. Skipping MLflow logging.")
except Exception as e:
    print(f" Could not log to MLflow: {e}")


print(" Evaluation completed successfully!")
