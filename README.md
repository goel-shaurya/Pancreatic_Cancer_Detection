# Pancreatic Cancer Detection Project

## Overview
- **Objective**: Develop a predictive model to detect pancreatic cancer using various machine learning algorithms.
- **Dataset**: Contains patient data with features relevant for cancer detection.

## Data Features
- **Features**:
  - Demographic and clinical information about patients.
  - Biochemical markers and imaging data related to pancreatic cancer.

## Methodology
- **Data Preprocessing**:
  - Handle missing values and outliers.
  - Normalize and scale features.
- **Exploratory Data Analysis (EDA)**:
  - Analyze distributions, correlations, and relationships between features.
  - Identify patterns and insights from data.
- **Modeling**:
  - **Models Used**:
    - Logistic Regression
    - LightGBM
    - Random Forest Classifier
    - Extra Trees Classifier
    - XGBoost
    - Support Vector Machine (SVM)
    - K-Nearest Neighbors (KNN)
    - Decision Tree Classifier
    - AdaBoost Classifier
  - **Evaluation Metrics**:
    - Accuracy
    - Precision
    - Recall
    - F1-Score

## Results

| Model                        | Accuracy | Precision | Recall | F1-Score |
|------------------------------|----------|-----------|--------|----------|
| **Logistic Regression**      | 72.82%   | 0.73      | 0.74   | 0.74     |
| **LightGBM**                 | 87.69%   | 0.88      | 0.88   | 0.88     |
| **Random Forest Classifier** | 86.67%   | 0.87      | 0.87   | 0.87     |
| **Extra Trees Classifier**   | 88.21%   | 0.89      | 0.89   | 0.89     |
| **XGBoost**                  | 86.67%   | 0.87      | 0.87   | 0.87     |
| **Support Vector Machine (SVM)** | 50.26%   | 0.50      | 0.51   | 0.48     |
| **K-Nearest Neighbors (KNN)**| 65.64%   | 0.67      | 0.66   | 0.66     |
| **Decision Tree Classifier** | 86.67%   | 0.87      | 0.87   | 0.87     |
| **AdaBoost Classifier**      | 83.08%   | 0.84      | 0.84   | 0.84     |

## Installation
- **Dependencies**:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `seaborn`
  - `matplotlib`
  - `plotly`
  - `lightgbm`
  - `xgboost`
  - `yellowbrick`


## Contribution
- Contributions are welcome. Please submit a pull request with detailed descriptions of changes or improvements.
