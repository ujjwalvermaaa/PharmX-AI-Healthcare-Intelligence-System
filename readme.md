# 🧠 PharmX AI – Healthcare Intelligence System

> **End-to-End AI-Powered Healthcare Analytics & Multi-Model Prediction Platform**

---

## 🔍 Project Overview

**PharmX AI** is a comprehensive **healthcare intelligence system** that integrates multiple machine learning models into a unified platform for disease prediction, hospital analysis, outbreak forecasting, and healthcare demand estimation.

The project follows a **complete industry-level pipeline**, starting from raw healthcare datasets to fully deployed interactive dashboards using Streamlit.

---

## 🎯 Problem Statement

Healthcare systems today face major challenges:

* Fragmented healthcare data across multiple domains
* Delayed disease detection
* Poor hospital resource planning
* Lack of predictive insights for outbreaks and demand
* No integration of structured + unstructured data

These issues lead to inefficient decision-making and poor patient outcomes.

---

## 💡 Solution

PharmX solves this by building a **multi-model AI system** that:

* Predicts diseases using patient vitals & symptoms
* Detects diseases from text (NLP)
* Estimates hospital severity
* Predicts outbreak risk
* Forecasts medicine sales
* Provides interactive visual analytics

---

## 🏗️ End-to-End Pipeline

Raw Data → Cleaning → Feature Engineering → Encoding → Model Training → Evaluation → Deployment → Streamlit App

---

## 🧩 Core Modules

### 🧠 Disease Prediction

Predicts disease using vitals, lifestyle, and symptoms.

Multiple machine learning algorithms were trained and evaluated, including:

* Support Vector Machine (SVM)
* XGBoost
* Gradient Boosting
* Logistic Regression
* Random Forest
* K-Nearest Neighbors (KNN)

After comparative evaluation, **XGBoost achieved the highest accuracy and best overall performance**, and was therefore selected as the final model for deployment.

---

### 💬 NLP Disease Detection

Predicts disease from user text input using TF-IDF + ML.

---

### 🏥 Hospital Severity

Estimates hospitalization risk and severity levels.

---

### 🌍 Outbreak Risk

Predicts outbreak probability using environmental and case data.

---

### 📈 Medicine Demand Prediction

Predicts pharmacy sales using demand indicators.

---

## 📁 Project Folder Structure

PHARMX/  
│  
├── data/  
│   ├── raw/  
│   │   ├── 1_patient_health_records.csv  
│   │   ├── 2_hospital_admissions.csv  
│   │   ├── 3_pharmacy_sales.csv  
│   │   ├── 4_doctor_prescriptions.csv  
│   │   ├── 5_weather_data.csv  
│   │   ├── 6_population_demographics.csv  
│   │   ├── 7_disease_outbreak.csv  
│   │   ├── 8_health_tweets.csv  
│   │   ├── 9_medicine_info.csv  
│   │   └── DataPipeline.ipynb  
│  
│   ├── processed/  
│   │   ├── patient_clean.csv  
│   │   ├── hospital_clean.csv  
│   │   ├── outbreak_clean.csv  
│   │   ├── medicine_clean.csv  
│   │   ├── pharmacy_clean.csv  
│   │   ├── population_clean.csv  
│   │   ├── prescription_clean.csv  
│   │   ├── tweets_clean.csv  
│   │   └── weather_clean.csv  
│  
│   ├── feature_engineered/  
│   │   ├── patient_features.csv  
│   │   ├── hospital_features.csv  
│   │   ├── outbreak_features.csv  
│   │   ├── pharmacy_features.csv  
│   │   └── tweets_features.csv  
│  
│   └── encoded/  
│       ├── patient_encoded.csv  
│       ├── hospital_encoded.csv  
│       ├── outbreak_encoded.csv  
│       ├── medicine_encoded.csv  
│       ├── pharmacy_encoded.csv  
│       ├── population_encoded.csv  
│       ├── prescription_encoded.csv  
│       ├── tweets_encoded.csv  
│       └── weather_encoded.csv  
│  
├── models/  
│   ├── disease_prediction_model.pkl  
│   ├── disease_encoder.pkl  
│   ├── hospital_model.pkl  
│   ├── outbreak_model.pkl  
│   ├── medicine_model.pkl  
│   ├── health_risk_model.pkl  
│   ├── nlp_model.pkl  
│   ├── tfidf.pkl  
│   ├── scaler.pkl  
│   ├── sales_scaler.pkl  
│   └── pharmacy_sales_dl_model.h5  
│  
├── notebooks/  
│   ├── 01_data_understanding.ipynb  
│   ├── 02_data_cleaning.ipynb  
│   ├── 03_feature_engineering.ipynb  
│   ├── 04_encoding_and_feature_preparation.ipynb  
│   ├── 05_disease_prediction_models.ipynb  
│   ├── 06_model_evaluation_and_explainability.ipynb  
│   ├── 07_nlp_disease_detection.ipynb  
│   ├── 08_deep_learning_demand_prediction.ipynb  
│   └── 09_advanced_multi_model_health_analytics.ipynb  
│  
├── reports/  
│   ├── PharmXPresentation.pptx  
│   ├── PharmXPresentation.pdf  
│  
│   └── screenshots/  
│       ├── Disease/  
│       │   ├── 1.png  
│       │   ├── 2.png  
│       │   └── Full.png  
│  
│       ├── Disease(Text Prediction)/  
│       │   ├── 1.png  
│       │   └── 2.png  
│  
│       ├── Hospital Severity/  
│       │   ├── 1.png  
│       │   └── 2.png  
│  
│       ├── Outbreak Risk/  
│       │   ├── 1.png  
│       │   └── 2.png  
│  
│       ├── Overview/  
│       │   ├── 1.png  
│       │   ├── 2.png  
│       │   ├── 3.png  
│       │   └── Full.png  
│       
│       ├── Sales/  
│       │   ├── 1.png  
│       │   └── 2.png  
│  
│       └── Visualizations/  
│           ├── 1.png  
│           ├── 2.png  
│           ├── 3.png  
│           ├── 4.png  
│           ├── 5.png  
│           ├── 6.png  
│           ├── 7.png  
│           └── 8.png  
│  
├── src/  
│   ├── app.py  
│   ├── utils.py  
│   ├── bg.jpeg  
│   └── __pycache__/  
│  
├── requirements.txt  
└── README.md  

---

## 🧪 Data Pipeline Stages

### 1. Raw Data

* Multiple healthcare datasets (patients, hospitals, weather, etc.)

### 2. Processed Data

* Cleaned datasets with missing values handled

### 3. Feature Engineering

* Derived features like:
  * BMI
  * Risk scores
  * Aggregated metrics

### 4. Encoding

* Converted categorical variables into numeric format

---

## 🤖 Machine Learning Models

| Module             | Model Type     |
| ------------------ | -------------- |
| Disease Prediction | Classification |
| NLP Detection      | TF-IDF + ML    |
| Hospital Severity  | Regression     |
| Outbreak Risk      | Classification |
| Medicine Demand    | Regression     |
| Sales Prediction   | Deep Learning  |

---

## 📊 Visualizations

* Disease distribution charts
* Risk score comparisons
* Sales trends
* Outbreak patterns

Built using **Plotly**

---

## 🚀 Deployment

Run locally:

pip install -r requirements.txt  
streamlit run src/app.py  

---

## ⚠️ Challenges Solved

* Feature mismatch errors
* Multi-model integration
* Consistent preprocessing pipeline

---

## 🔮 Future Scope

* Explainable AI (SHAP)
* Real-time API deployment
* Cloud integration
* Mobile app

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* TensorFlow
* NLP/NLTK (TF-IDF)
* Deep Learning (ANN)
* Plotly
* Streamlit

---

## 👨‍💻 Author

**Ujjwal Verma**

---

## ⭐ Final Note

PharmX AI represents a **complete real-world AI system**, showcasing:

* End-to-end ML pipeline
* Multi-domain data integration
* Deployment-ready architecture
* Strong problem-solving skills
