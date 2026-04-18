import pandas as pd
import shap
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
from dotenv import load_dotenv


load_dotenv()

# Set MLflow tracking from environment variables
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")

# Set MLflow tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Create reports directory
os.makedirs("reports", exist_ok=True)

# Load test data
X_test = pd.read_csv("data/processed/X_test.csv")

# Load best model from MLflow
client = MlflowClient()
experiment = client.get_experiment_by_name("Default")

# Handle case where experiment might not exist
if experiment is None:
    print("Experiment 'Default' not found. Creating it...")
    experiment = client.create_experiment("Default")

runs = client.search_runs(experiment.experiment_id)

# Handle case where no runs exist
if len(runs) == 0:
    print("No runs found in experiment. Please run training first.")
    exit(1)

best_run = sorted(
    runs,
    key=lambda x: x.data.metrics.get("roc_auc", 0),
    reverse=True
)[0]

best_run_id = best_run.info.run_id
print("Best run ID:", best_run_id)

model_uri = f"runs:/{best_run_id}/model"
best_model = mlflow.sklearn.load_model(model_uri)

# SHAP analysis
print("Computing SHAP values...")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)

# Create SHAP plot
print("Creating SHAP summary plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values, X_test, show=False)
plt.tight_layout()
plt.savefig("reports/shap_summary.png", bbox_inches='tight', dpi=150)
plt.close()

print("SHAP plot saved to reports/shap_summary.png")

# Log SHAP plot to MLflow as artifact
with mlflow.start_run(run_name="explainability", nested=True):
    # Log the SHAP plot image
    mlflow.log_artifact("reports/shap_summary.png")
    
    # Log SHAP values as a text file 
    import json
    shap_summary = {
        "shape": shap_values.shape if hasattr(shap_values, 'shape') else "list",
        "features": list(X_test.columns),
        "n_samples": len(X_test)
    }
    
    with open("reports/shap_summary.json", "w") as f:
        json.dump(shap_summary, f, indent=2)
    
    mlflow.log_artifact("reports/shap_summary.json")
    
    print("SHAP plot and metadata logged to MLflow")

print("Explainability stage completed successfully!")