# 📊 Telco Customer Churn Prediction

This project predicts **customer churn** using the [Telco Customer Churn dataset](https://www.kaggle.com/blastchar/telco-customer-churn).  
It was developed as a practical end-to-end machine learning exercise to explore preprocessing, model selection, and threshold tuning.

---

## 🚀 Project Overview

The goal is to predict whether a customer will **churn** (leave the company) based on their service usage and demographics.  
Particular emphasis was placed on **minimizing false negatives (FN)** — that is, correctly identifying customers who are likely to churn.

---

## 🧠 Main Steps

### 1. Data Loading & Cleaning
- Missing values were found in `TotalCharges` and coerced to numeric.
- Duplicates and inconsistent entries were removed.

### 2. Preprocessing Pipeline
Built a modular `Pipeline` using **scikit-learn**:
- **Numerical features:** imputed with the median and scaled using `StandardScaler`.
- **Categorical features:** encoded with `OneHotEncoder(drop='first')`.
- **Binary features:** passed through *without transformation* (using `"passthrough"`) since they were already in numeric 0/1 format.

This ensures consistent preprocessing for both training and inference.

### 3. Model Training and Comparison
Three models were tested using 5-fold cross-validation:
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|-----------|------------|---------|-----|----------|
| Logistic Regression | 0.804 | 0.664 | 0.532 | 0.591 | 0.849 |
| Random Forest | 0.789 | 0.632 | 0.492 | 0.553 | 0.824 |
| HistGradientBoosting | 0.794 | 0.634 | 0.524 | 0.573 | 0.835 |

🔹 The **HistGradientBoostingClassifier** (HGB) was selected as the final model due to its strong performance and robustness.

### 4. Regularization & Learning Curves
A regularized version of HGB was tested (`l2_regularization`, reduced tree depth, smaller learning rate).

Although it slightly reduced overfitting (smaller train–validation gap),  
its overall performance dropped slightly:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|--------|-----------|------------|---------|-----|----------|
| Original HGB | 0.81 | 0.67 | 0.52 | 0.59 | 0.845 |
| Regularized HGB | 0.80 | 0.65 | 0.52 | 0.58 | 0.844 |

✅ Conclusion: Regularization wasn’t necessary since the original model was already generalizing well.

### 5. Threshold Tuning
Since recall was the main business priority, the decision threshold was adjusted:

| Threshold | Precision | Recall | F1 | Accuracy |
|------------|------------|---------|------------|-----------|
| 0.5 (default) | 0.65 | 0.52 | 0.58 | 0.80 |
| 0.3 (custom)  | 0.54 | 0.77 | 0.63 | 0.76 |

Lowering the threshold from **0.5 → 0.3** increased recall significantly —  
catching more actual churners at the cost of some false positives, which aligns with the project goal.

---

## 📈 Key Insights

- **Recall** was prioritized over precision to minimize false negatives (missed churners).  
- **Learning curves** confirmed good generalization with minimal overfitting.  
- **Feature importance** helped interpret which customer attributes most influenced churn.  
- A custom threshold improved the balance between recall and precision for business use.

---

## 📂 Repository Structure

```bash
telco-customer-churn/
├── data/
│ ├── telco-customer-churn.zip
│ └── extracted/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│ └── best_hgb_model.pkl
│
├── notebooks/
│ └── telco_customer_churn.ipynb
│
├── requirements.txt
└── README.md

## 🧩 Requirements

To reproduce the notebook, install dependencies with:

```bash
pip install -r requirements.txt