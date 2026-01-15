from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def get_model_candidates(task_name=None):
    if task_name == 'heart':
        return {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
    elif task_name == 'diabetes':
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000)
        }
    else:
        # Fallback or default behavior if needed, currently returning both for safety/legacy support if no task specified
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000),
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
        }
