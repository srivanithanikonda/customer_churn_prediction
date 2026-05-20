# Customer Churn Prediction using Machine Learning

> **MBA Project — DY Patil University Online, Navi Mumbai**  
> Program: MBA in Data Science and Business Analytics

---

# Table of Contents

1. Introduction  
2. Background  
3. Objectives  
4. Methodology  
5. Dataset  
6. Project Structure  
7. Results  
8. Key Insights  
9. How to Run  
10. Technologies Used  
11. Conclusions & Recommendations  
12. Limitations  
13. References  

---

# Introduction

Customer churn refers to customers discontinuing their relationship with a company.  
In the telecommunications industry, customer retention is highly important because retaining existing customers is generally more cost-effective than acquiring new ones.

This project uses machine learning techniques to predict whether a customer is likely to churn based on demographic, service, and billing information. The goal is to help telecom companies identify high-risk customers early and take proactive retention actions.

---

# Background

The telecommunications industry faces intense competition, resulting in frequent customer switching between service providers.

This project analyses customer behavior patterns and develops predictive models to classify customers as:

- Likely to Churn
- Likely to Stay

The project also identifies the major factors influencing churn so businesses can improve customer retention strategies.

---

# Objectives

- Predict customer churn using machine learning algorithms
- Compare multiple classification models
- Identify the most important factors affecting churn
- Generate visual insights using data analysis
- Provide business recommendations for customer retention

---

# Methodology

## 1. Data Collection

- Source: IBM Watson Analytics Telco Customer Churn Dataset
- Dataset obtained from Kaggle
- Total records: ~7,043 customers
- Total features: 21 columns

---

## 2. Data Pre-processing

The following preprocessing steps were performed:

- Converted `TotalCharges` from string to numeric format
- Handled missing values by removing incomplete records
- Dropped non-predictive column (`customerID`)
- Encoded target variable:
  - Yes → 1
  - No → 0
- Applied one-hot encoding using `pd.get_dummies(drop_first=True)`

---

## 3. Exploratory Data Analysis (EDA)

Several visualisations were created to analyse customer behavior patterns:

- Customer churn distribution
- Gender vs churn
- Contract type vs churn
- Internet service vs churn
- Monthly charges vs churn
- Tenure distribution by churn status

These visualisations help identify customer groups with higher churn probability.

---

## 4. Model Building

The dataset was divided into training and testing sets using an 80:20 split.

Three machine learning classification models were trained and evaluated:

| Model | Description |
|---|---|
| Logistic Regression | Linear classification model |
| Decision Tree | Tree-based classification model |
| Random Forest | Ensemble learning model |

---

## 5. Model Evaluation

The models were evaluated using:

- Accuracy Score
- Confusion Matrix
- Classification Report
- Precision
- Recall
- F1-Score
- Feature Importance Analysis

---

# Dataset

| Feature | Description |

| `tenure` | Number of months customer stayed |
| `MonthlyCharges` | Monthly bill amount |
| `TotalCharges` | Total amount charged |
| `Contract` | Contract type |
| `PaymentMethod` | Customer payment method |
| `InternetService` | Type of internet service |
| `Churn` | Target variable |

---

# Project Structure

Customer_Churn_Project/

 -main.py
 - README.md
 - requirements.txt
 -WA_Fn-UseC_-Telco-Customer-Churn.csv

- graphs/
    ├── churn_distribution.png
    ├── gender_vs_churn.png
    ├── contract_vs_churn.png
    ├── internet_service_vs_churn.png
    ├── monthly_charges_vs_churn.png
    ├── tenure_distribution.png
    ├── model_accuracy_comparison.png
    └── feature_importance.png


---

# Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 78.53% |
| Decision Tree | 72.49% |
| Random Forest | 78.53% |

Both Logistic Regression and Random Forest achieved similar performance.  
Random Forest additionally provided feature importance insights for business analysis.

---

# Key Insights

- Customers with month-to-month contracts show higher churn rates
- Customers with higher monthly charges are more likely to churn
- New customers with low tenure have higher churn probability
- Fiber optic internet users show comparatively higher churn
- Customers with long-term contracts show significantly lower churn rates

---

# Graphs Generated

The project automatically generates and saves visualisations inside the `graphs/` folder during execution.

These visualisations support business decision-making by highlighting customer segments with high churn probability.

---

# How to Run

## Prerequisites

- Python 3.8 or above
- pip

---

## Steps


# Created virtual environment
python -m venv .venv

# Activated virtual environment (Windows)
.venv\Scripts\activate

# Installed dependencies
pip install -r requirements.txt

#had Run the project
python main.py
```

All generated graphs will be saved automatically in the `graphs/` folder.

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming Language |
| pandas | Data manipulation |
| matplotlib | Data visualisation |
| seaborn | Statistical visualisation |
| scikit-learn | Machine learning |

---

# Conclusions & Recommendations

## Conclusions

- Logistic Regression and Random Forest achieved the best performance
- Contract type, monthly charges, and tenure strongly influence churn
- Machine learning can effectively identify customers at risk of leaving

---

## Recommendations

1. Target month-to-month customers with long-term contract offers
2. Introduce loyalty benefits for new customers
3. Improve retention strategies for high-billing customers
4. Use predictive models within CRM systems for proactive retention
5. Conduct periodic retraining of the model using updated customer data

---

# Limitations

- Dataset is collected from  US telecom customers only.
- External economic and seasonal factors were not included
- Imbalanced data may affect prediction quality
- Model performance may vary with real-world deployment

Future improvements :

- SMOTE oversampling to balance the datasheet
- Hyperparameter tuning for better model optimization 
- Advanced ensemble techniques to improve accuracy

---

# References

1. IBM Watson Analytics — Telco Customer Churn Dataset  
   https://www.kaggle.com/blastchar/telco-customer-churn

2. Scikit-learn Documentation  
   https://scikit-learn.org

3. Pandas Documentation  
   https://pandas.pydata.org

4. Provost, F., & Fawcett, T. (2013). *Data Science for Business*. O'Reilly Media.

5. Han, J., Kamber, M., & Pei, J. (2011). *Data Mining: Concepts and Techniques*.

---

# Academic Declaration

This project was developed as part of the MBA program in Data Science and Business Analytics at DY Patil University Online, Navi Mumbai.
## Author
Sri Vani