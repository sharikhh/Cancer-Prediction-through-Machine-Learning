# 🧬 Cancer Cell Prediction Using Machine Learning

A machine learning project to predict whether a tumor is **benign** or **malignant** using diagnostic data from cell samples.

---

## 📌 Table of Contents

- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Dataset Overview](#dataset-overview)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Technologies Used](#technologies-used)
- [Results](#results)
- [Team Members](#team-members)
- [License](#license)

---

## 🧾 Introduction

Cancer is a disease in which some of the body’s cells grow uncontrollably and may spread to other parts of the body. It develops when the body's normal control mechanism stops functioning — old cells don’t die and instead grow out of control, forming abnormal cells and possibly tumors.

---

## ❓ Problem Statement

The goal of this project is to analyze tumor-related features to predict whether a tumor is **benign (non-cancerous)** or **malignant (cancerous)** using machine learning algorithms.

---

## 🗃️ Dataset Overview

The dataset includes diagnostic features of tumor cells. Each row in the dataset contains:

| Feature       | Description                        |
|---------------|------------------------------------|
| ID            | Unique identifier                  |
| Clump         | Clump ID                           |
| Clump Thickness | Thickness of cell clumps         |
| UnifSize      | Uniformity of cell size            |
| UnifShape     | Uniformity of cell shape           |
| MargAdh       | Marginal adhesion                  |
| SingEpiSize   | Single epithelial cell size        |
| BareNuc       | Bare nuclei                        |
| BlandChrom    | Bland chromatin                    |
| NormNucl      | Normal nucleoli                    |
| Mit           | Mitoses                            |
| Class         | Target: Benign (2) or Malignant (4)|

📊 **Source:** [UCI Machine Learning Repository – Breast Cancer Dataset](https://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+(original))

---

## 🧠 Machine Learning Pipeline

> A step-by-step process used to build the classification model:

1. **Data Cleaning & Preprocessing**
   - Handling missing/null values
   - Label encoding for categorical values
   - Feature scaling for better model performance

2. **Exploratory Data Analysis (EDA)**
   - Visualizations to understand feature distributions
   - Correlation heatmaps and boxplots

3. **Model Building**
   - Applied classification algorithms like:
     - Logistic Regression
     - Decision Trees
     - Random Forest
     - Support Vector Machine (SVM)

4. **Evaluation**
   - Confusion matrix
   - Accuracy, Precision, Recall, F1-Score
   - Cross-validation

---

## 🧰 Technologies Used

- Python 🐍
- NumPy & Pandas
- Matplotlib & Seaborn
- Scikit-learn
- Jupyter Notebook

---

## 📈 Results

Each model was evaluated using standard metrics. The model with the best balance of **accuracy** and **generalization** was selected for final prediction.

> *You can update this section later with performance scores and visual results.*  

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
