
# Machine Learning Zoomcamp (2026) - Homework 2: Linear Regression

This repository contains my solution for **Homework 2** of the [Machine Learning Zoomcamp 2026](https://github.com/DataTalksClub/machine-learning-zoomcamp) cohort. The focus of this assignment is building, regularizing, and evaluating a linear regression model from scratch using NumPy and Pandas.

---

## 📌 Project Overview
The objective of this homework is to predict car fuel efficiency (`fuel_efficiency_mpg`) based on key physical and engine specifications. 

### Features Used:
- `engine_displacement`
- `horsepower` *(contains missing values)*
- `vehicle_weight`
- `model_year`
- **Target Variable:** `fuel_efficiency_mpg`

---

## 🛠️ Key Steps & Methodology

1. **Exploratory Data Analysis (EDA) & Missing Value Imputation:**
   - Identified missing values in the `horsepower` column and calculated baseline statistics (e.g., median horsepower of `254`).
   - Compared two imputation strategies: filling missing values with `0` versus filling them with the **training set mean**. Found that imputing with `0` yielded a slightly better validation RMSE.

2. **Model Training & Regularization:**
   - Implemented standard linear regression using normal equations (`(X^T X)^{-1} X^T y`).
   - Experimented with regularized linear regression (Ridge regression) across various regularization parameters (`r` values: `[0, 0.01, 0.1, 1, 5, 10, 100]`).

3. **Model Stability Analysis:**
   - Tested model stability across multiple random split seeds (`0` to `9`) with a 60%/20%/20% train/validation/test split distribution.
   - Calculated the standard deviation of validation RMSE scores to evaluate model robustness.

4. **Final Evaluation:**
   - Trained the final regularized model on the combined training and validation sets using an optimal regularization parameter and evaluated performance on the unseen test dataset.

---

## 🚀 How to Run the Code

Make sure you have `numpy` and `pandas` installed, then run the solution script:

```bash
python homework_02.py
