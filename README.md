# 🧠 Diabetes Prediction — End-to-End Machine Learning Project

An end-to-end **Machine Learning pipeline** that predicts whether a patient is diabetic based on medical diagnostic measurements.

This project demonstrates **data preprocessing, model training, evaluation, hyperparameter tuning, and deployment using Streamlit**.

---

## 🚀 Project Overview

Diabetes is a chronic medical condition that requires early detection for effective management.

This project:

* Cleans and preprocesses medical data
* Trains multiple ML models
* Optimizes for medical recall (minimizing false negatives)
* Saves trained models
* Deploys an interactive prediction app using Streamlit

---

## 📊 Dataset Features

The dataset contains the following features:

* Pregnancies
* Glucose
* BloodPressure
* SkinThickness
* Insulin
* BMI
* DiabetesPedigreeFunction
* Age
* Outcome (Target: 0 = Non-Diabetic, 1 = Diabetic)

---

## 🧪 Machine Learning Pipeline

### 1️⃣ Data Cleaning

* Replaced invalid zero values with median
* Standardized features using `StandardScaler`

### 2️⃣ Model Training

Models tested:

* Logistic Regression
* Random Forest (Primary Model)
* Gradient Boosting

### 3️⃣ Hyperparameter Tuning

Used `GridSearchCV` optimizing for **Recall**, which is critical in medical prediction to reduce false negatives.

### 4️⃣ Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

---

## 🎯 Why Recall Optimization?

In medical diagnosis:

> False Negative = Dangerous

Predicting a diabetic patient as non-diabetic can delay treatment.

Therefore, this project prioritizes **Recall** to minimize risk.

---

## 📈 Feature Importance

The most influential features typically include:

* Glucose
* BMI
* Age

This provides interpretability and medical insight.

---

## 🛠️ Tech Stack

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Streamlit

---

## 📌 Future Improvements

* Add SHAP for explainability
* Deploy using FastAPI
* Dockerize the application
* Add CI/CD pipeline
* Integrate MLflow for experiment tracking

---

## 👨‍💻 Author

**Arnav Purohit**
Machine Learning Enthusiast | Aspiring SWE
