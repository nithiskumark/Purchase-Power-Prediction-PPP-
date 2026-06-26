# PurchaseIQ: Purchasing Power Prediction (PPP)

> **An end-to-end Machine Learning project that predicts an individual's purchasing power using demographic and financial indicators.**

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)

---

# Overview

PurchaseIQ is a Machine Learning project that predicts an individual's **Purchasing Power** using financial and demographic features.

The project demonstrates a complete industry-standard Machine Learning workflow including:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- Experiment Tracking with MLflow
- Model Registry
- Model Deployment Ready Pipeline

The objective is to estimate whether a customer has **Low**, **Medium**, or **High Purchasing Power**, helping businesses make better financial and marketing decisions.

---

# Business Problem

Financial institutions and businesses often need to estimate the purchasing capability of customers before offering products, loans, premium memberships, or targeted advertisements.

Traditional rule-based systems cannot accurately capture customer financial behavior.

This project uses Machine Learning to automatically predict purchasing power based on customer financial characteristics.

---

# Dataset

Each record represents one customer.

| Feature | Description |
|----------|-------------|
| Age | Customer age |
| Income | Annual Income |
| Savings Rate | Percentage of income saved |
| Debt Ratio | Debt divided by income |
| Purchasing Power | Target Variable |

Example

| Age | Income | Savings Rate | Debt Ratio | Purchasing Power |
|------|---------|--------------|------------|-----------------|
|25|45000|0.18|0.30|Medium|
|42|98000|0.35|0.15|High|
|31|28000|0.05|0.60|Low|

---

# Project Workflow

```
Dataset
   │
   ▼
Data Cleaning
   │
   ▼
Exploratory Data Analysis
   │
   ▼
Feature Engineering
   │
   ▼
Train/Test Split
   │
   ▼
Model Training
   │
   ▼
Hyperparameter Tuning
   │
   ▼
Model Evaluation
   │
   ▼
MLflow Experiment Tracking
   │
   ▼
Model Registry
   │
   ▼
Deployment
```

---

# Features Used

- Age
- Income
- Savings Rate
- Debt Ratio

Target

- Purchasing Power

---

# Machine Learning Pipeline

The pipeline includes:

- Missing Value Handling
- Feature Scaling
- Label Encoding (if required)
- Train-Test Split
- Logistic Regression Model
- GridSearchCV
- Model Evaluation
- MLflow Logging
- Model Registration

---

# Technologies Used

| Category | Tools |
|-----------|------|
| Language | Python |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn |
| Experiment Tracking | MLflow |

---


---

# Sample Prediction

Input

```text
Age            : 32
Income         : 60000
Savings Rate   : 0.25
Debt Ratio     : 0.20
```

Prediction

```text
Purchasing Power : High
```

# MLflow Tracking

The project logs

✔ Parameters

- Regularization Strength
- Solver
- Penalty
- Max Iterations

✔ Metrics

- Accuracy
- Precision
- Recall
- F1 Score

✔ Artifacts

- Confusion Matrix
- Classification Report
- Feature Importance
- Trained Model

✔ Model Registry

- Registered Model
- Versioning
- Stage Transition

---

# Business Applications

- Credit Risk Assessment
- Loan Eligibility Prediction
- Customer Segmentation
- Financial Planning
- Personalized Marketing
- Retail Analytics
- Banking Recommendation Systems
- Insurance Premium Analysis

---

# Future Improvements

- XGBoost
- LightGBM
- CatBoost
- Random Forest
- Explainable AI (SHAP)
- Streamlit Dashboard
- Docker Deployment
- CI/CD Pipeline
- AWS Deployment
- Kubernetes
- Model Monitoring

---


# Results

The Logistic Regression model successfully predicts customer purchasing power using financial indicators with high accuracy after hyperparameter tuning and proper preprocessing.

The complete ML lifecycle is managed using MLflow, making the project reproducible, scalable, and production-ready.

---

# Learning Outcomes

Through this project, you will learn:

- Data Cleaning
- EDA
- Feature Engineering
- Machine Learning Pipelines
- Logistic Regression
- Hyperparameter Tuning
- GridSearchCV
- Model Evaluation
- MLflow Experiment Tracking
- Model Registry
- Model Deployment
- Production-ready ML Workflow

---

# Author

**Nithis Kumar K**

Data Scientist | Machine Learning Engineer | AI Enthusiast


---

---

## ⭐ If you found this project useful, consider giving it a Star on GitHub!
