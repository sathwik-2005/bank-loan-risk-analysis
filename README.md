# Bank Loan Risk Analysis

A machine learning project that predicts loan approval status based on applicant financial and demographic data, and identifies the key factors driving approval decisions.

## Problem Statement

Banks need to assess loan applications quickly and consistently. This project builds a classification model to predict whether a loan application will be approved, and analyzes which applicant attributes matter most in that decision.

## Dataset

- **Source**: [Loan Prediction Problem Dataset](https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset) (Kaggle)
- **Size**: 614 rows, 13 columns
- **Features**: Gender, Marital Status, Dependents, Education, Self-Employment status, Applicant/Co-applicant Income, Loan Amount, Loan Term, Credit History, Property Area
- **Target**: Loan_Status (Approved / Not Approved)

## Approach

1. **Data Cleaning**: Handled missing values — categorical columns filled with mode, numerical columns filled with median/mode as appropriate
2. **Exploratory Data Analysis**: Visualized loan approval rates across Credit History, Education, Property Area, and Marital Status
3. **Feature Engineering**: Encoded categorical variables using Label Encoding
4. **Modeling**: Trained and compared two classifiers:
   - Logistic Regression (baseline)
   - Random Forest Classifier
5. **Evaluation**: Assessed both models using accuracy, precision, recall, and confusion matrices
6. **Interpretation**: Extracted feature importance from the Random Forest model to explain key drivers of approval

## Key Findings

- **Credit History is the single strongest predictor** of loan approval — both the EDA and Random Forest feature importance confirm this. Applicants with a good credit history are approved at a dramatically higher rate.
- Applicant Income and Loan Amount are the next most influential features.
- Logistic Regression (78.9% accuracy) slightly outperformed Random Forest (75.6% accuracy) on this dataset, showing that simpler models can be competitive on small, well-structured data.

## Results

| Model | Accuracy |
|---|---|
| Logistic Regression | 78.9% |
| Random Forest | 75.6% |

## Tools Used

Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn

## Project Structure
```
bank-loan-risk/
├── data/
│   └── train.csv
├── notebooks/
│   └── bank_loan_risk_analysis.ipynb
└── README.md
```
