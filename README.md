# 💰 HealthCare Cost Analysis

This project predicts individual health insurance charges based on personal and lifestyle factors such as age, BMI, smoking status, number of dependents, and region. By applying regression models to medical insurance datasets, we aim to build an accurate and interpretable system that helps insurers and policyholders understand cost drivers and estimate future premiums.

---

## 🎯 Project Objectives

- Predict medical insurance charges using supervised regression models.
- Identify key features (e.g., smoking, age, BMI) that influence insurance costs.
- Evaluate and compare model performance using error metrics (MAE, MSE, R²).
- Visualize relationships and distributions to enhance model interpretability.


---

## 🔍 Dataset Overview

- **Source**: [Medical Cost Personal Dataset – Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Rows**: 1338
- **Features**:
  - `age`: Age of primary beneficiary
  - `sex`: Insurance contractor gender
  - `bmi`: Body Mass Index
  - `children`: Number of dependents covered
  - `smoker`: Smoking status
  - `region`: Residential region in the U.S.
  - `charges`: Medical insurance cost (target variable)

---

## 🛠️ Tech Stack

- **Python**, **NumPy**, **Pandas**, **Matplotlib**, **Seaborn**, **Scikit-learn**
- Regression Models: Linear Regression, Polynomial Regression, Regularized models
- Model evaluation metrics: R² Score, MAE, MSE, RMSE

---

## 📊 Key Features & Highlights

- 📌 **Exploratory Data Analysis (EDA)**:
  - Heatmaps, pairplots, and distribution plots to uncover trends and outliers
  - Correlation analysis to identify influential predictors

- 📈 **Modeling & Evaluation**:
  - Multiple regression models trained and compared
  - Regularization (Ridge, Lasso) for bias-variance trade-off
  - Error metrics reported to evaluate real-world performance

- 📉 **Feature Impact**:
  - Smoking emerged as a strong predictor of increased costs
  - BMI and age showed non-linear influence, motivating polynomial features

---

## 🚀 How to Run

1. Clone or download the repository.
2. Open `Healthcarecostanalysis.ipynb` in Jupyter or VS Code.
3. Run all cells sequentially to reproduce EDA, modeling, and predictions.

---

## 📈 Sample Visualizations (Optional if you add screenshots)

| Visualization             | Insight                                                |
|--------------------------|--------------------------------------------------------|
| Charges vs Smoking       | Smokers pay 2x+ more on average                        |
| Charges vs Age           | Strong positive correlation, especially for smokers    |
| Heatmap of Correlations  | Charges highly correlated with smoking and age         |
| Residual Plots           | Reveals overfitting or underfitting in model behavior  |

---

## 🧠 Insights

- **Smoking status** has the strongest influence on insurance charges.
- **Age** and **BMI** show significant non-linear patterns.
- Predictive models like polynomial regression improved accuracy over plain linear models.
- This model could support better underwriting decisions and transparent pricing structures.

---

## 📌 Future Enhancements

- Integrate hyperparameter tuning with `GridSearchCV`
- Add XGBoost/Gradient Boosted Trees for advanced regression
- Deploy as a web app using Streamlit or Flask for real-time predictions
