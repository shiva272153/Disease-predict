import os
import json
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from joblib import load
import csv
from datetime import datetime

app = Flask(__name__)
app.secret_key = "change_this_for_prod"

BASE_DIR = os.path.dirname(__file__)
ARTIFACTS = os.path.join(BASE_DIR, 'models', 'artifacts')

def load_model_scaler(task):
    model_path = os.path.join(ARTIFACTS, f"{task}_model.pkl")
    scaler_path = os.path.join(ARTIFACTS, f"{task}_scaler.pkl")
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        return None, None
    return load(model_path), load(scaler_path)

@app.route("/")
def index():
    # Metrics display removed in favor of simple prediction focus
    return render_template("index.html", metrics={})

@app.route("/predict/heart", methods=["GET", "POST"])
def predict_heart():
    # Define expected feature order (must match training dataset columns excluding 'target')
    feature_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
    ]

    if request.method == "POST":
        try:
            patient_name = request.form.get("patient_name", "Patient")
            values = []
            for name in feature_names:
                v = request.form.get(name)
                if v is None or v == "":
                    raise ValueError(f"Missing value for {name}")
                values.append(float(v))
            X = np.array(values).reshape(1, -1)

            model, scaler = load_model_scaler('heart')
            if model is None:
                flash("Model not trained yet. Please run training.", "danger")
                return redirect(url_for('index'))

            X_scaled = scaler.transform(X)
            # Convert back to DataFrame to preserve feature names and avoid warning
            X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
            
            proba = None
            try:
                proba = float(model.predict_proba(X_scaled_df)[0, 1])
            except Exception:
                # fallback to decision function or predicted label
                proba = None
            pred = int(model.predict(X_scaled_df)[0])

            # Save prediction to CSV
            try:
                log_file = os.path.join(BASE_DIR, 'data', 'logged_predictions.csv')
                file_exists = os.path.isfile(log_file)
                
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        # Write header
                        header = ['timestamp'] + feature_names + ['prediction', 'probability']
                        writer.writerow(header)
                    
                    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + values + [pred, proba if proba is not None else ""]
                    writer.writerow(row)
            except Exception as log_err:
                print(f"Failed to log prediction: {log_err}")

            return render_template(
                "result.html",
                name=patient_name,
                disease_name="Heart Disease",
                prediction=pred,
                probability=proba,
                interpretation="High risk" if pred == 1 else "Low risk",
                back_url=url_for('predict_heart')
            )
        except Exception as e:
            flash(f"Error: {e}", "danger")
            return redirect(url_for('predict_heart'))

    return render_template("predict_heart.html")

@app.route("/predict/diabetes", methods=["GET", "POST"])
def predict_diabetes():
    feature_names = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
    ]

    if request.method == "POST":
        try:
            patient_name = request.form.get("patient_name", "Patient")
            values = []
            for name in feature_names:
                v = request.form.get(name)
                # Handle potential missing values safely if needed, or assume required
                if v is None or v == "":
                    raise ValueError(f"Missing value for {name}")
                values.append(float(v))
            
            X = np.array(values).reshape(1, -1)

            model, scaler = load_model_scaler('diabetes')
            if model is None:
                flash("Model not trained yet. Please run training.", "danger")
                return redirect(url_for('index'))

            X_scaled = scaler.transform(X)
            # Convert back to DataFrame to preserve feature names for the model
            X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)

            proba = None
            try:
                # Some models like linear svc might not have predict_proba by default unless calibrated or specific param
                # LogReg has it by default
                proba = float(model.predict_proba(X_scaled_df)[0, 1])
            except Exception:
                proba = None
            
            pred = int(model.predict(X_scaled_df)[0])

            # Save prediction to CSV
            try:
                log_file = os.path.join(BASE_DIR, 'data', 'logged_predictions.csv')
                file_exists = os.path.isfile(log_file)
                
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        header = ['timestamp', 'type'] + feature_names + ['prediction', 'probability']
                        writer.writerow(header)
                    
                    # Note: We are adding 'type' column to distinguish, but previously we didn't have it.
                    # Adapting to just append or maybe we should have separate log files?
                    # For simplicity, let's just append to the same file but we might have schema mismatch.
                    # User requested explanation earlier implies "logged_predictions.csv" is general.
                    # Let's write to "logged_predictions_diabetes.csv" to avoid schema conflict or just accept it.
                    # I will use a separate file for safety: logged_predictions_diabetes.csv
                    pass 
                
                # Actually, let's just log to a specific diabetes log file
                diabetes_log = os.path.join(BASE_DIR, 'data', 'logged_predictions_diabetes.csv')
                d_exists = os.path.isfile(diabetes_log)
                with open(diabetes_log, 'a', newline='') as f:
                    writer = csv.writer(f)
                    if not d_exists:
                         writer.writerow(['timestamp'] + feature_names + ['prediction', 'probability'])
                    
                    row = [datetime.now().strftime("%Y-%m-%d %H:%M:%S")] + values + [pred, proba if proba is not None else ""]
                    writer.writerow(row)

            except Exception as log_err:
                print(f"Failed to log prediction: {log_err}")

            return render_template(
                "result.html",
                name=patient_name,
                disease_name="Diabetes",
                prediction=pred,
                probability=proba,
                interpretation="Positive" if pred == 1 else "Negative",
                back_url=url_for('predict_diabetes')
            )
        except Exception as e:
            flash(f"Error: {e}", "danger")
            return redirect(url_for('predict_diabetes'))

    return render_template("predict_diabetes.html")

@app.route("/documentation")
def documentation():
    return render_template("documentation.html")

@app.route("/privacy-policy")
def privacy_policy():
    return render_template("privacy_policy.html")

@app.route("/terms-of-service")
def terms_of_service():
    return render_template("terms_of_service.html")

if __name__ == "__main__":
    app.run(debug=True)
