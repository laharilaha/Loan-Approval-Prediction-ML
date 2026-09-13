# Loan Approval Prediction using Machine Learning

## Project Overview

Loan approval is an important process in banks and financial institutions. This project develops a machine learning-based system that predicts whether a loan application is likely to be **Approved** or **Rejected** based on applicant information.

The system uses the **Random Forest Classifier** as the main machine learning algorithm and provides an interactive **Streamlit web application** for making predictions.

The application also uses **SHAP (SHapley Additive exPlanations)** to identify the important factors that influence an individual prediction.

---

## Problem Statement

Traditional loan approval processes may require the analysis of several applicant details manually. This can be time-consuming when a large number of applications need to be evaluated.

The objective of this project is to develop a machine learning system that can analyse applicant information and predict the possible loan approval status efficiently.

---

## Objectives

- Predict whether a loan application will be **Approved or Rejected**.
- Implement the **Random Forest Classifier**.
- Preprocess numerical and categorical applicant information.
- Evaluate the performance of the trained model.
- Provide prediction confidence to the user.
- Explain individual predictions using **SHAP**.
- Develop an interactive web application using **Streamlit**.

---

## Algorithm Used

### Random Forest Classifier

Random Forest is an ensemble machine learning algorithm that combines multiple Decision Trees to produce a final prediction.

In this project:

- **Algorithm:** Random Forest Classifier
- **Number of Trees:** 150
- **Random State:** 42
- **Class Weight:** Balanced
- **Classification:** Approved / Rejected
- **Test Accuracy:** Approximately **98.01%**

---

## Dataset

The project uses the file:

```text
loan_approval_dataset.csv