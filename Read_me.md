# Energy Efficiency Prediction Using Machine Learning

## Overview

This project predicts **Heating Load** and **Cooling Load** of residential buildings using **multi-output regression models**.
The goal is to demonstrate end-to-end machine learning workflow including **exploratory data analysis, feature scaling, model comparison, neural networks, and model persistence**.

The problem is treated as a **regression task with two continuous target variables**, which closely resembles real-world energy optimization and sustainability use cases.

---

## Dataset Description

* **Dataset**: Energy Efficiency Dataset
* **Total Records**: 768
* **Features**: 8 numerical building characteristics
* **Targets**:

  * `Heating_Load`
  * `Cooling_Load`

### Input Features

* Relative Compactness
* Surface Area
* Wall Area
* Roof Area
* Overall Height
* Orientation
* Glazing Area
* Glazing Area Distribution

### Data Quality

* No missing values
* No duplicate records
* All features are numeric

---

## Exploratory Data Analysis

* Descriptive statistics for all variables
* Distribution analysis using histograms
* Outlier detection using boxplots
* Target variable correlation inspection

---

## Data Preprocessing

* Train-test split (80/20)
* Feature scaling using **StandardScaler**
* Multi-output target handling for regression models

---

## Models Implemented

### Linear Regression

* Baseline multi-output regression model

**Performance**:

* Heating Load R² ≈ 0.91
* Cooling Load R² ≈ 0.89

---

### Ridge Regression (Regularized)

* Implemented using `MultiOutputRegressor`
* GridSearchCV used to tune alpha parameter

**Best Alpha**: 0.1
**Cross-validated R²**: ≈ 0.89

---

### Lasso Regression

* Feature-shrinking regularized regression
* Comparable performance to Ridge

---

### Neural Network (Deep Learning)

* Fully connected feed-forward neural network
* Two output neurons for multi-target prediction

**Architecture**:

* Dense (64, ReLU)
* Dense (32, ReLU)
* Dense (2 outputs)

**Results**:

* Heating Load R² ≈ 0.96
* Cooling Load R² ≈ 0.93
* Best overall performance among all models

---

## Model Evaluation Metrics

* Mean Squared Error (MSE)
* R² Score (coefficient of determination)
* Model-wise comparison for both targets

---

## Sample Prediction

The trained neural network can predict both heating and cooling loads for new building configurations using the saved scaler and model.

---

## Model Persistence

* Feature scaler saved using `joblib`
* Neural network saved as `.h5` file for reuse and deployment

---

## Project Structure

```
energy-efficiency-prediction/
│
├── energy_efficiency.ipynb
├── nn_energy_model.h5
├── scaler.joblib
├── requirements.txt
└── README.md
```

---

## Tech Stack

* Python
* Pandas, NumPy
* Matplotlib, Seaborn
* Scikit-learn
* TensorFlow / Keras
* Joblib

---

## Key Takeaways

* Multi-output regression effectively models related energy targets
* Neural networks significantly outperform linear models for this task
* Proper feature scaling is critical for neural network performance
* End-to-end ML pipelines enable easy deployment

---

## Potential Applications

* Building energy optimization
* Sustainable architecture planning
* Smart city energy modeling
* Green building certification support

---

## Future Enhancements

* Hyperparameter tuning for neural networks
* Feature importance and sensitivity analysis
* Model explainability using SHAP
* Web or API-based deployment for predictions

---

## Author

**Jai Kishan Kokkiligadda**
Data Science and Machine Learning

---
* Suggest **deployment architecture** (API or cloud)

