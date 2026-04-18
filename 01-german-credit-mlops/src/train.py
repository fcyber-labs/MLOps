import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import yaml
import os
from sklearn.model_selection import train_test_split, GridSearchCV  
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from mlflow.models import infer_signature
from urllib.parse import urlparse
from dotenv import load_dotenv
import warnings
import json
import matplotlib.pyplot as plt

load_dotenv()
warnings.filterwarnings('ignore')

# Load parameters 
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

train_params = params["train"]

# Set MLflow tracking URI 
os.environ["MLFLOW_TRACKING_URI"] = os.getenv("MLFLOW_TRACKING_URI")
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

# Hyperparameter tuning
def hyperparameter_tuning(X_train, y_train, param_grid, cv_folds=5):
    """Perform hyperparameter tuning with GridSearchCV"""
    rf = RandomForestClassifier(random_state=train_params["random_state"])
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=cv_folds, 
        n_jobs=-1, 
        verbose=2,
        scoring='roc_auc'  
    )
    grid_search.fit(X_train, y_train)
    return grid_search

def train(data_path, model_path, random_state):
    # Load data
    df = pd.read_csv(data_path)
    
    target_col = 'Risk_bad' if 'Risk_bad' in df.columns else 'Risk_good'
    
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state
    )
    
    # Save test sets
    os.makedirs("data/processed", exist_ok=True)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)
    
    # Define hyperparameter grid
    param_grid = {
        "max_depth": [3, 5, 7, 10, None],
        "n_estimators": [3, 5, 10, 25, 50, 150],
        "max_features": [4, 7, 15, 20]
    }
    
    print("Starting hyperparameter tuning...")
    print(f"Parameter grid: {param_grid}")
    total_combinations = len(param_grid['max_depth']) * len(param_grid['n_estimators']) * len(param_grid['max_features'])
    print(f"Total combinations to try: {total_combinations}")
    
    # Perform hyperparameter tuning
    grid_search = hyperparameter_tuning(X_train, y_train, param_grid)
    
    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print("Hyperparameter tuning completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best cross-validation ROC-AUC score: {best_score:.4f}")
    
    # Make predictions with best model
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    roc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTest set performance:")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Calculate confusion matrix and classification report
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    # Create models directory and save model locally FIRST
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(best_model, model_path)
    print(f"Model saved to {model_path}")
    
    # Start MLflow run to log all comparisons
    with mlflow.start_run(run_name="rf_hyperparameter_tuning"):
        # Log parameters
        mlflow.log_param("best_max_depth", best_params["max_depth"])
        mlflow.log_param("best_n_estimators", best_params["n_estimators"])
        mlflow.log_param("best_max_features", best_params["max_features"])
        mlflow.log_param("random_state", random_state)
        
        # Log metrics
        mlflow.log_metric("best_cv_roc_auc", best_score)
        mlflow.log_metric("test_roc_auc", roc)
        mlflow.log_metric("test_accuracy", accuracy)
        
        # Log artifacts
        print("\nLogging artifacts to MLflow...")
        
        mlflow.log_text(str(cm), "confusion_matrix.txt")
        print("  confusion_matrix.txt")
        
        mlflow.log_text(cr, "classification_report.txt")
        print("  classification_report.txt")
        
        # Log grid search results
        results_df = pd.DataFrame(grid_search.cv_results_)
        results_df.to_csv("grid_search_results.csv", index=False)
        mlflow.log_artifact("grid_search_results.csv")
        print("  grid_search_results.csv")
        
        # Log model file
        mlflow.log_artifact(model_path)
        print("  model.pkl")
        
        # Log feature importance plot
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'][:15], feature_importance['importance'][:15])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importances - Random Forest')
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact("feature_importance.png")
        print("  feature_importance.png")
        
        # Log parameters summary
        params_summary = {
            "best_params": best_params,
            "best_cv_score": float(best_score),
            "test_roc_auc": float(roc),
            "test_accuracy": float(accuracy),
            "n_features": X_train.shape[1],
            "n_train_samples": X_train.shape[0],
            "n_test_samples": X_test.shape[0]
        }
        with open("params_summary.json", "w") as f:
            json.dump(params_summary, f, indent=2)
        mlflow.log_artifact("params_summary.json")
        print("  params_summary.json")
        
        # Log the model with signature
        signature = infer_signature(X_train, best_model.predict(X_train))
        mlflow.sklearn.log_model(best_model, "best_model", signature=signature)
        
        print("All artifacts logged to MLflow!")
    
    # Return results
    return {
        "best_params": best_params,
        "best_cv_score": best_score,
        "test_roc_auc": roc,
        "test_accuracy": accuracy
    }

if __name__ == "__main__":
    results = train(
        data_path=train_params["data"],
        model_path=train_params["model"],
        random_state=train_params["random_state"]
    )
    
    print("\nFinal Results Summary")
    print(f"Best Parameters: {results['best_params']}")
    print(f"Best CV ROC-AUC: {results['best_cv_score']:.4f}")
    print(f"Test ROC-AUC: {results['test_roc_auc']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")