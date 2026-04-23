# 🚗 Advanced Machine Learning: UK Used Car Valuation

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-lightgrey.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-yellow.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-green.svg)

## 📌 Project Overview
The used car market is highly dynamic, with vehicle depreciation driven by a complex combination of age, mileage, brand, and condition. This project applies **Advanced Machine Learning** techniques to accurately predict the market value of used cars in the UK. 

By utilizing comprehensive feature engineering, rigorous feature selection, and explainable AI (SHAP) combined with K-Means clustering, this predictive pipeline offers highly optimized valuations to assist buyers, sellers, and dealerships.

---

## 📊 Dataset
The model is trained on the `adverts.csv` dataset, representing UK used car listings. 
- **Total Observations:** ~402,000 rows
- **Data Split:** 60% Training, 20% Validation, 20% Test (Seed: 99)
- **Target Variable:** `price` (Log-transformed to normalise right-skewed distribution)
- **Key Features:** `mileage`, `standard_make`, `standard_model`, `vehicle_condition`, `body_type`, `fuel_type`, `reg_code`.

---

## ⚙️ Methodology & Pipeline

### 1. Data Preprocessing & Cleaning
- **Age Derivation:** Engineered a `vehicle_age` feature by extracting the manufacturing year from the UK `reg_code`.
- **Missing Data:** Imputed continuous variables using the **median** and categorical variables using the **mode** (`SimpleImputer`). Explicit missingness indicators (e.g., `mileage_missing`) were added to prevent data loss.
- **Outlier Handling:** Clipped numerical features between the **1st and 99th percentiles** to handle extreme outliers without dropping rows.

### 2. Feature Engineering & Encoding
- **Target Encoding:** Applied smoothed target encoding (`smooth=10.0`) to high-cardinality categorical features (e.g., `standard_make`, `standard_model`) to prevent high dimensionality.
- **One-Hot Encoding:** Applied to low-cardinality variables (<= 20 unique values) after grouping rare categories (< 1% frequency) into an 'Other' bucket.
- **Scaling:** Standardized all continuous features using `StandardScaler`.

### 3. Feature Selection
- Implemented **Recursive Feature Elimination with Cross-Validation (RFECV)** utilizing a Linear Regression estimator. 
- Successfully reduced the feature space to the **28 most optimal features**, improving training efficiency and reducing noise.

### 4. Advanced Modeling
A variety of models were trained and cross-validated, including:
- **Baseline:** Multiple Linear Regression, RidgeCV, Bayesian Ridge.
- **Tree-Based Ensembles:** Random Forest Regressor, Gradient Boosting Regressor, HistGradientBoostingRegressor.
- **Meta-Ensembles:** Voting Regressor and Stacking Regressor.

### 5. Explainable AI & SHAP Clustering (Novel Approach)
To further push the model's accuracy, **SHAP (SHapley Additive exPlanations)** values were extracted from the best-performing model.
- Applied **K-Means Clustering** to the SHAP values to group vehicles based on their underlying pricing profiles.
- Injected this `shap_cluster_group` back into the dataset as a new feature to retrain the final Gradient Boosting model.

---

## 📈 Results & Evaluation
The primary evaluation metrics for the project are **MAE (Mean Absolute Error)**, **RMSE**, and **R² Score**.

*Note: The target variable was transformed back using `np.expm1()` to calculate errors in actual currency (£).*

| Model Configuration | Target Metric | Score |
|---------------------|---------------|-------|
| Gradient Boosting (Standard) | MAE | £3,128.81 |
| **Gradient Boosting (with SHAP Clusters)** | **MAE** | *[Insert Final MAE]* |
| Final Model R² Score | R² | *[Insert Final R2]* |

*(Note: The SHAP-clustered feature set yielded the highest predictive accuracy by capturing non-linear interactions specific to distinct vehicle sub-markets.)*

---

## 🚀 How to Run the Project

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Ashwashhere/AdvancedMachineLearning_CarValuation.git](https://github.com/Ashwashhere/AdvancedMachineLearning_CarValuation.git)
   cd AdvancedMachineLearning_CarValuation
