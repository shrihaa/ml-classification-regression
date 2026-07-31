# 🤖 Predictive Maintenance & Housing Price Prediction

Two end-to-end machine learning projects built with Python and scikit-learn: a binary classification model predicting factory machine failure, and a regression model predicting housing prices.

---

## 🛠️ Tech Stack

- **Python**
- **scikit-learn** — modeling, preprocessing, hyperparameter tuning
- **Pandas** — data cleaning and preprocessing
- **Matplotlib** & **Seaborn** — data visualization

---

## 🏭 Part A — Predicting Factory Machine Status (Classification)

**Task:** Binary classification — predict whether a machine is Working (0) or Faulty (1) based on sensor readings.

**Process:**
- Preprocessed data with `OneHotEncoder` (categorical features) and `StandardScaler` (numerical features)
- Imputed missing values (median for numerical, mode for categorical)
- Trained and compared **Logistic Regression**, **Random Forest**, and **KNN**
- Evaluated models using confusion matrices, F1-score, and ROC-AUC
- Tuned the best model (Random Forest) with `RandomizedSearchCV`
- Analyzed feature importance to identify key failure predictors

**Results:**
- Dummy baseline F1-score: 0.07
- Tuned Random Forest F1-score: **0.87** (12x improvement)
- **Torque** identified as the strongest predictor of machine failure — higher torque correlates with higher failure likelihood

---

## 🏠 Part B — Predicting Housing Price (Regression)

**Task:** Regression — predict continuous housing prices based on property features.

**Process:**
- Preprocessed data with `OneHotEncoder` and `StandardScaler`
- Trained and compared **Linear Regression**, **Random Forest**, **Ridge Regression**, and **Lasso Regression**
- Evaluated models using RMSE, MSE, and R²
- Selected **Lasso Regression** as the final model for its automatic feature selection (L1 regularization)
- Tuned alpha using `GridSearchCV`

**Results:**
- Lasso Regression significantly outperformed the dummy baseline on RMSE, MSE, and R²
- Strongest price drivers: **house area**, **number of toilets**, and **renovation status**
- Irrelevant features (e.g. `city_seattle`) automatically eliminated via L1 regularization

---

## 📁 Contents

- `part_a_classification_machine_failure.ipynb` — full classification workflow
- `part_b_regression_housing_price.ipynb` — full regression workflow

---

## 👤 Author

**Shriha**
