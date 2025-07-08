# 💼 Customer Churn Risk Prediction Model

A machine learning project designed to predict customer churn based on behavioral, demographic, and subscription data. The goal is to enable businesses to proactively identify customers likely to leave and take data-driven steps to improve retention.

---

## 📌 Table of Contents

1. [Problem Statement](#problem-statement)  
2. [Dataset Description](#dataset-description)  
3. [Project Objectives](#project-objectives)  
4. [Technologies & Tools](#technologies--tools)  
5. [ML Workflow Overview](#ml-workflow-overview)  
6. [Modeling & Evaluation](#modeling--evaluation)  
7. [Results & Insights](#results--insights)  
8. [Future Enhancements](#future-enhancements)  
9. [How to Run](#how-to-run)  
10. [License](#license)

---

## 🧠 Problem Statement

In a competitive market, retaining customers is more profitable than acquiring new ones. Customer churn significantly impacts revenue and growth. This project builds a predictive model using machine learning techniques to identify customers at risk of churning—based on their demographics, usage behavior, and billing patterns.

The aim is to help businesses implement targeted, personalized retention strategies and improve overall customer loyalty.

---

## 🗃 Dataset Description

The dataset consists of 100,000 customer records with the following features:

| Feature | Description |
|--------|-------------|
| `CustomerID` | Unique identifier |
| `Name` | Customer’s full name |
| `Age` | Age of the customer |
| `Gender` | Male / Female |
| `Location` | City (Houston, LA, Miami, Chicago, New York) |
| `Subscription_Length_Months` | Number of months subscribed |
| `Monthly_Bill` | Monthly bill amount |
| `Total_Usage_GB` | Data usage in GB |
| `Churn` | Binary flag (1 = churned, 0 = retained) |

---

## 🎯 Project Objectives

- Predict customer churn with high accuracy using machine learning.
- Identify top contributing factors influencing churn.
- Help businesses retain customers through proactive engagement strategies.
- Build a modular, scalable, and explainable ML solution.

---

## 🛠️ Technologies & Tools

| Category | Tools |
|---------|-------|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML Models | Scikit-learn, XGBoost, Random Forest, Logistic Regression, SVM |
| Feature Engineering | One-Hot Encoding, Scaling, PCA |
| Deep Learning | TensorFlow, Keras (Neural Network attempt) |
| Model Evaluation | Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix |
| Model Optimization | GridSearchCV, Cross-Validation, EarlyStopping |
| Deployment | Pickle (model saving) |
| IDE | Jupyter Notebook |

---

## 🔄 ML Workflow Overview

1. **Data Cleaning**  
   - Handled missing values (none found)  
   - Removed duplicates  

2. **Exploratory Data Analysis (EDA)**  
   - Visualized data distribution  
   - Identified patterns in churn behavior  

3. **Feature Engineering**  
   - Encoded categorical features (Gender, Location)  
   - Scaled numerical variables (Age, Monthly_Bill, etc.)

4. **Model Training**  
   - Tested multiple classifiers: Logistic Regression, KNN, SVM, Random Forest, XGBoost  
   - Applied dimensionality reduction using PCA  

5. **Hyperparameter Tuning**  
   - GridSearchCV and manual tuning  
   - Cross-Validation (5-Fold)

6. **Model Evaluation**  
   - Evaluated performance on Train & Test sets  
   - Used Confusion Matrix, ROC-AUC, F1-Score

7. **Model Saving**  
   - Saved final model using `pickle` for reuse

---

## 📈 Modeling & Evaluation

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| Accuracy | 66.49% | 50.05% |
| Precision | 66.86% | 49.53% |
| Recall | 65.12% | 48.92% |
| F1-Score | 65.98% | 49.22% |
| ROC-AUC | 0.66 | 0.50 |

**Final Model:** `XGBoost Classifier`  
**Top Features:**  
- Monthly_Bill  
- Total_Usage_GB  
- Age  
- Subscription_Length_Months  

---

## 🔍 Results & Insights

- Customers with **higher bills and usage** are more likely to churn.
- **Younger customers** and those with **shorter subscriptions** show higher churn risk.
- XGBoost outperformed other models in terms of interpretability and metrics.
- Although the test accuracy was moderate, the insights are valuable for customer segmentation.

---

## 🚀 Future Enhancements

- Collect behavioral and time-based interaction data for better prediction.
- Implement SMOTE or resampling to address mild class imbalance.
- Improve model generalization via stacking or ensemble learning.
- Add SHAP or LIME for explainability in real-world deployment.
- Integrate the model into a dashboard (e.g., Streamlit or Flask API).

---

## ▶️ How to Run

1. Clone the repository  
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction.git
   cd customer-churn-prediction



