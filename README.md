# 🧾 Loan Application Prediction – EDA & Classification

This repository contains two Jupyter notebooks focused on predicting the approval status of loan applications. The workflow is divided into two phases: data preprocessing and exploration, and model training and evaluation. This project demonstrates a complete data science pipeline from raw data to predictive modeling.

---

## 📘 About the Notebooks

## ✅ Phase 1: Data Preprocessing & Exploration

In the first phase, we focus on understanding the structure and quality of the dataset. Various data cleaning steps are performed, including handling missing values, treating outliers, and analyzing relationships between variables. The aim is to prepare a refined dataset ready for model training.

This notebook covers:
- Loading and exploring the loan dataset
- Identifying and treating outliers
- Analyzing correlations among variables
- Handling missing values
- Encoding categorical features
- Normalizing and preparing the data for modeling

## 🤖 Phase 2: Model Building & Evaluation

The second phase involves building predictive models using various classification techniques. Feature selection is performed to reduce dimensionality, and models are evaluated using performance metrics such as AUC and accuracy. This phase also includes neural network development and hyperparameter tuning.

This notebook focuses on:
- Feature selection using mutual information
- Training multiple classification models:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Random Forest
- Naive Bayes
- Neural Network (using Keras)

Hyperparameter tuning with GridSearchCV

Evaluating models using ROC-AUC, accuracy, and classification metrics

Neural network tuning on:

- Number of units
- Activation functions
- Dropout rates
- Batch size and learning rate decay

Statistical comparison of model performance using paired t-tests

---

## 📁 Files Required
Before running the notebooks, ensure you have the following files:

- `Phase1.csv` – Dataset used in the preprocessing phase.
- `Phase2.csv` – Cleaned and processed data used for training models.

These should be located in the same directory as the notebooks.

---

## 🧠 Model Evaluation Metrics
The performance of all models is evaluated based on:

- ROC-AUC Score
- Accuracy
- Classification Report
- Confusion Matrix

Neural networks are additionally fine-tuned for optimal performance through layered hyperparameter tuning.

---

## ▶️ How to Run
1. Clone the repository:
  ```bash
  https://github.com/JayZxD/Loan-Application-Prediction-.git
  ```
2. Make sure the following Python libraries are installed:

  - numpy
  - pandas
  - matplotlib
  - seaborn
  - scikit-learn
  - tensorflow / keras

3. Run the notebooks:

  - `Phase1.ipynb`
  - `Phase2.ipynb`

⚠️ Disclaimer
Some methods and techniques in these notebooks were adapted from academic coursework for learning purposes. No real customer data is used. This is a synthetic or anonymized dataset and should be used solely for educational demonstrations.

## 📬 Author
[Jay Mayekar](https://www.linkedin.com/in/jay-mayekar25/)

🤝 Contributors
[Arya Patil](https://www.linkedin.com/in/aryapatil/), [Rohan Nagansur](https://www.linkedin.com/in/rohannagansur/)
