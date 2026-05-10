# Student Performance Prediction with Hybrid Machine Learning Models

Predicting student academic performance using supervised machine learning, behavioral clustering, and hybrid modeling techniques.

## Overview

This project explores whether behavioral segmentation using K-Means clustering can improve the prediction of student exam scores compared to traditional supervised learning models.

The project builds a complete machine learning pipeline using:

- Data preprocessing
- Supervised regression models
- K-Means clustering
- Hybrid learning approaches
- Model evaluation and visualization

The dataset contains **20,000 student records** with demographic, academic behavior, lifestyle, and learning environment features.

The main research question:

> Can behavioral clustering improve the prediction of student exam scores compared to supervised learning models trained only on the original features?

---

# Features

- Reproducible preprocessing pipeline
- Supervised regression baselines
- K-Means behavioral clustering
- Hybrid supervised + clustering models
- PCA cluster visualization
- Performance comparison charts
- Exported evaluation CSV files

---

# Dataset

The dataset includes the following features:

| Feature | Type |
|---|---|
| student_id | Identifier |
| age | Numeric |
| gender | Categorical |
| course | Categorical |
| study_hours | Numeric |
| class_attendance | Numeric |
| internet_access | Categorical |
| sleep_hours | Numeric |
| sleep_quality | Categorical |
| study_method | Categorical |
| facility_rating | Categorical |
| exam_difficulty | Categorical |
| exam_score | Target |

### Target Variable
- `exam_score`

### Removed Feature
- `student_id` was removed because it is non-predictive.

---

# Preprocessing Pipeline

The dataset was split into:

- **80% Training**
- **20% Testing**

To prevent data leakage:

- All preprocessing transformations were fit only on training data.
- Test data used the fitted preprocessing pipeline.

### Numeric Features
- Median imputation
- StandardScaler normalization

### Categorical Features
- Most frequent imputation
- OneHotEncoder encoding
- Unknown categories ignored during testing

---

# Models Used

## Supervised Regression Models

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor

## Unsupervised Learning

- K-Means Clustering

Cluster labels were later added as an additional feature in the hybrid models.

---

# K-Means Clustering

Candidate values of `k` from 2 to 8 were evaluated using:

- Elbow Method
- Silhouette Score

### Best Cluster Count
- `k = 2`

### Cluster Evaluation

| k | Inertia | Silhouette Score |
|---|---|---|
| 2 | 124386.8284 | 0.0851 |
| 3 | 117062.2145 | 0.0732 |
| 4 | 111260.0582 | 0.0761 |
| 5 | 106807.3824 | 0.0736 |
| 6 | 103603.9511 | 0.0690 |
| 7 | 100554.1322 | 0.0715 |
| 8 | 97548.9561 | 0.0752 |

The low silhouette scores indicate weak cluster separation in the feature space.

---

# Results

## Baseline Model Performance

| Model | RMSE | R² Score |
|---|---|---|
| Linear Regression | 9.7725 | 0.7330 |
| Gradient Boosting | 9.8503 | 0.7287 |
| XGBoost | 9.8845 | 0.7269 |
| Random Forest | 10.2904 | 0.7040 |

## Hybrid Model Performance (with K-Means Clusters)

| Model | RMSE | R² Score |
|---|---|---|
| Linear Regression | 9.7724 | 0.7330 |
| Gradient Boosting | 9.8503 | 0.7287 |
| XGBoost | 9.8637 | 0.7280 |
| Random Forest | 10.2978 | 0.7035 |

---

# Key Findings

- Linear Regression achieved the best overall performance.
- Adding K-Means cluster labels produced only minimal improvements.
- XGBoost showed the largest improvement:
  - `+0.001146 R²`
- Random Forest performance slightly decreased after adding clusters.
- Behavioral clustering is more useful for:
  - Interpretability
  - Exploratory analysis
  - Student segmentation
- Clustering did **not** significantly improve predictive accuracy in this dataset.

---

# Visualizations

The notebook includes:

- Elbow Method Plot
- Silhouette Score Plot
- PCA Cluster Visualization
- Baseline vs Hybrid R² Comparison
- Prediction vs Actual Scatter Plot

---

# Limitations

- Dataset may be synthetic or simulated
- Weak cluster separation
- No cross-validation
- Limited hyperparameter tuning
- No advanced interpretability methods (SHAP, permutation importance)

---

# Future Work

Potential future improvements:

- K-Fold Cross Validation
- Hyperparameter tuning
- SHAP value analysis
- Feature engineering
- Gaussian Mixture Models
- Hierarchical clustering
- Real institutional datasets
- Temporal academic features
- GPA and assignment history integration

---

# Reproducibility

The notebook exports:

- `final_model_results.csv`
- `kmeans_cluster_evaluation.csv`

These files allow the reported results to be reproduced and validated.

---

# Repository

https://github.com/therealsolo2/intro-to-ds-final

---

# Installation

Clone the repository:

```bash
git clone https://github.com/therealsolo2/intro-to-ds-final.git
cd intro-to-ds-final
```

Run the notebook:

```bash
jupyter notebook
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- Jupyter Notebook
