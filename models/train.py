import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import dump
from registry import get_model_candidates
from evaluators import evaluate_model, cross_validate_model
from preprocessors import preprocess_heart, preprocess_diabetes

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(ARTIFACT_DIR, exist_ok=True)

def train_for_task(task_name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, preprocess_fn):
    print(f"=== Training for {task_name} ===")
    
    # Preprocess both
    prep_train = preprocess_fn(train_df)
    prep_test = preprocess_fn(test_df)
    
    # Train and evaluate all models
    models = get_model_candidates(task_name)
    best_score = -1.0
    best_name = None
    best_model = None
    task_metrics = {}

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(prep_train.X, prep_train.y)
        score = model.score(prep_test.X, prep_test.y)
        print(f"  {name} Accuracy: {score:.3f}")
        task_metrics[name] = score
        
        if score > best_score:
            best_score = score
            best_name = name
            best_model = model

    print(f"Best model for {task_name}: {best_name} with accuracy {best_score:.3f}")

    # Persist best model and scaler
    dump(best_model, os.path.join(ARTIFACT_DIR, f"{task_name}_model.pkl"))
    dump(prep_train.scaler, os.path.join(ARTIFACT_DIR, f"{task_name}_scaler.pkl"))
    print(f"Saved {best_name} for {task_name}")

    return task_metrics, best_name

def main():
    metrics_all = {}

    # Heart
    heart_train_path = os.path.join(DATA_DIR, 'heart_train.csv')
    heart_test_path = os.path.join(DATA_DIR, 'heart_test.csv')
    
    if os.path.exists(heart_train_path) and os.path.exists(heart_test_path):
        heart_train = pd.read_csv(heart_train_path)
        heart_test = pd.read_csv(heart_test_path)
        heart_metrics, heart_best = train_for_task('heart', heart_train, heart_test, preprocess_heart)
        metrics_all['heart'] = {"best": heart_best, "models": heart_metrics}

    # Diabetes
    diabetes_train_path = os.path.join(DATA_DIR, 'diabetes_train.csv')
    diabetes_test_path = os.path.join(DATA_DIR, 'diabetes_test.csv')
    
    if os.path.exists(diabetes_train_path) and os.path.exists(diabetes_test_path):
        diabetes_train = pd.read_csv(diabetes_train_path)
        diabetes_test = pd.read_csv(diabetes_test_path)
        diabetes_metrics, diabetes_best = train_for_task('diabetes', diabetes_train, diabetes_test, preprocess_diabetes)
        metrics_all['diabetes'] = {"best": diabetes_best, "models": diabetes_metrics}

    # Save metrics
    with open(os.path.join(ARTIFACT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics_all, f, indent=2)

if __name__ == "__main__":
    main()
