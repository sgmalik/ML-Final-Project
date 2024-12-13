---
marp: true
title: Housing Price Prediction
theme: default
paginate: true
header: "ML Final Project: Rent Prediction"
footer: "By: Surya Malik and Cierra Church"
---

# Housing Price Prediction

## By: Surya Malik and Cierra Church

---

# Objective

### **Goal**:
- Predict housing prices based on features such as:
  - Area, bedrooms, bathrooms, and more.

### **Why is this important?**
- **For Renters**: Ensure fair property pricing.
- **For Landlords/Homeowners**: Price properties competitively.

### **Challenges**
- **Limited Dataset**: Only 545 samples.
- **Overfitting Risk**: Especially for complex models.

---

# Dataset Overview

### **Features**
1. Area of the house (numerical)
2. Number of bedrooms, bathrooms, and stories (numerical)
3. Main road access, guestroom, basement (boolean)
4. Hot water heating, air conditioning (boolean)
5. Parking spots (numerical)
6. Furnishing status (categorical → one-hot encoded)

### **Target Variable**
- **Price**: Continuous variable.

---

# Preprocessing

### **Steps Taken**
1. **Handle Categorical Data**:
   - One-hot encoding for `furnishingstatus`.
2. **Normalize Features**:
   - Used `StandardScaler` for all numerical features.
3. **Split Data**:
   - **80% training**, **20% testing** split.

---

# Models Implemented

1. **Linear Regression**
2. **Polynomial Regression**
3. **Ridge and Lasso Regression**
4. **Decision Tree Regressor**
5. **Random Forest Regressor**
6. **Logistic Regression (for classification)**
7. **Support Vector Machines**
8. **Neural Networks**

---

# Linear and Polynomial Regression

### **Linear Regression**
- R²: **0.68**
- Moderate performance.
- Struggles with non-linear relationships.

### **Polynomial Regression**
- Degree: **2**
- R²: **0.79**
- Better fit but **risk of overfitting**.

---

# Ridge and Lasso Regression

### **Why Regularization?**
- Reduces overfitting by penalizing large coefficients.

### **Results**
- Ridge: R² = **0.69**
- Lasso: R² = **0.69**

### **Conclusion**
- Slight improvement, but no significant generalization gains.

---

# Tree-Based Models

### **Decision Tree Regressor**
- Accuracy: **0.20** (basic)
- After tuning: **0.62**

### **Random Forest Regressor**
- Default Accuracy: **0.56**
- After tuning: **0.63**

### **Key Observations**
- **Area** was the most important feature in regressors.

---

# Transition to Classification

### **Why Switch to Classification?**
- Regression struggled with overfitting.
- Classification simplifies the problem.

### **Classes**
- Low, Medium, High, Premium.
- Used `pandas.cut()` to categorize prices.

---

# Classification Results

### **Logistic Regression**
- **Accuracy**: **93.58%**
- Severe overfitting despite regularization.

### **Support Vector Machines**
- Linear Kernel: **100% accuracy** → Overfit.
- Polynomial Kernel: **95.41% accuracy**.

---

# Decision Trees and Random Forests (Classification)

### **Decision Tree Classifier**
- Tuned Accuracy: **83.49%**
- Positive correlation with features.

### **Random Forest Classifier**
- **50 Trees**: Accuracy: **100%** → Overfit.

---

# Neural Networks

### Architectures Tried
1. **Simple 2-layer NN** → R²: -4.77 (worse than random).
2. **Improved 3-layer NN** → R²: 0.60.
3. **Complex 5-layer NN** → R²: 0.57 (overfitting).

---

# Evaluation Metrics

### **Regression Metrics**
1. Mean Absolute Error (MAE)
2. Mean Squared Error (MSE)
3. R² Score

### **Classification Metrics**
1. Accuracy
2. Precision, Recall, F1-score
3. Confusion Matrix

---

# Results Summary

| **Model**                | **R² / Accuracy** | **Notes**                     |
|---------------------------|-------------------|--------------------------------|
| Linear Regression         | R²: 0.68         | Moderate.                     |
| Polynomial Regression     | R²: 0.79         | Overfitting suspected.        |
| Ridge/Lasso Regression    | R²: 0.69         | Regularization improved fit.  |
| Decision Tree Regressor   | R²: 0.62         | Overfitting reduced w/ tuning.|
| Random Forest Regressor   | R²: 0.63         | Best regression model.        |
| Logistic Regression       | Acc: 93.58%      | Severe overfitting.           |
| SVM (Linear Kernel)       | Acc: 100%        | Overfit.                      |
| SVM (Poly Kernel)         | Acc: 95.41%      | Improved generalization.      |
| Random Forest Classifier  | Acc: 100%        | Overfit.                      |

---

# Feature Engineering

### **New Features Created**
1. **`area_per_bedroom`**: Area / Number of Bedrooms.
2. **`area_per_bathroom`**: Area / Number of Bathrooms.

### **Findings**
- Adding these features slightly decreased overfitting.
- Dropping irrelevant features also improved accuracy.

---

# Conclusions

1. **Best Model**:
   - **Regression**: Random Forest Regressor.
   - **Classification**: SVM (Polynomial Kernel).

2. **Overfitting Issues**:
   - Neural Networks and classification models overfit the small dataset.
   - Larger datasets are needed to validate generalization.

3. **Future Work**:
   - Collect more data.
   - Explore advanced feature engineering.
   - Consider ensemble techniques for regression.

---

# Questions?
Thank you!
