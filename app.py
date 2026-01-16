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
            
            # Validation helper
            def validate_range(val, name, min_v, max_v):
                if not (min_v <= val <= max_v):
                    raise ValueError(f"{name} must be between {min_v} and {max_v}")

            # Extract and validate
            # Age
            age = float(request.form.get("age"))
            validate_range(age, "Age", 1, 120)
            values.append(age)

            # Sex (0 or 1)
            sex = float(request.form.get("sex"))
            if sex not in [0, 1]: raise ValueError("Invalid Sex value")
            values.append(sex)

            # CP (0-3)
            cp = float(request.form.get("cp"))
            if cp not in [0, 1, 2, 3]: raise ValueError("Invalid Chest Pain Type")
            values.append(cp)

            # Trestbps (Resting BP)
            trestbps = float(request.form.get("trestbps"))
            validate_range(trestbps, "Resting BP", 50, 250)
            values.append(trestbps)

            # Chol
            chol = float(request.form.get("chol"))
            validate_range(chol, "Cholesterol", 100, 600)
            values.append(chol)

            # FBS (0 or 1)
            fbs = float(request.form.get("fbs"))
            if fbs not in [0, 1]: raise ValueError("Invalid Fasting Blood Sugar value")
            values.append(fbs)

            # RestECG (0-2)
            restecg = float(request.form.get("restecg"))
            if restecg not in [0, 1, 2]: raise ValueError("Invalid Resting ECG")
            values.append(restecg)

            # Thalach (Max Heart Rate)
            thalach = float(request.form.get("thalach"))
            validate_range(thalach, "Max Heart Rate", 30, 220)
            values.append(thalach)

            # Exang (0 or 1)
            exang = float(request.form.get("exang"))
            if exang not in [0, 1]: raise ValueError("Invalid Exercise Induced Angina value")
            values.append(exang)

            # Oldpeak
            oldpeak = float(request.form.get("oldpeak"))
            validate_range(oldpeak, "Oldpeak", 0, 10)
            values.append(oldpeak)

            # Slope (0-2)
            slope = float(request.form.get("slope"))
            if slope not in [0, 1, 2]: raise ValueError("Invalid ST Slope")
            values.append(slope)

            # CA (0-3)
            ca = float(request.form.get("ca"))
            if ca not in [0, 1, 2, 3]: raise ValueError("Invalid Major Vessels count")
            values.append(ca)

            # Thal (1-3)
            thal = float(request.form.get("thal"))
            if thal not in [1, 2, 3]: raise ValueError("Invalid Thalassemia value")
            values.append(thal)

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
        except ValueError as ve:
             flash(f"Input Error: {ve}", "danger")
             return redirect(url_for('predict_heart'))
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
            
            def validate_range(val, name, min_v, max_v):
                if not (min_v <= val <= max_v):
                    raise ValueError(f"{name} must be between {min_v} and {max_v}")

            # Pregnancies
            preg = float(request.form.get("Pregnancies"))
            validate_range(preg, "Pregnancies", 0, 20)
            values.append(preg)

            # Glucose
            gluc = float(request.form.get("Glucose"))
            validate_range(gluc, "Glucose", 50, 500)
            values.append(gluc)

            # BloodPressure
            bp = float(request.form.get("BloodPressure"))
            validate_range(bp, "BloodPressure", 40, 250)
            values.append(bp)

            # SkinThickness
            st = float(request.form.get("SkinThickness"))
            validate_range(st, "SkinThickness", 0, 100)
            values.append(st)

            # Insulin
            ins = float(request.form.get("Insulin"))
            validate_range(ins, "Insulin", 0, 900)
            values.append(ins)

            # BMI
            bmi = float(request.form.get("BMI"))
            validate_range(bmi, "BMI", 10, 70)
            values.append(bmi)

            # DPF
            dpf = float(request.form.get("DiabetesPedigreeFunction"))
            validate_range(dpf, "Diabetes Pedigree Function", 0.0, 3.0)
            values.append(dpf)

            # Age
            age = float(request.form.get("Age"))
            validate_range(age, "Age", 1, 120)
            values.append(age)
            
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
        except ValueError as ve:
            flash(f"Input Error: {ve}", "danger")
            return redirect(url_for('predict_diabetes'))
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
