# 🏏 Cricket Score Prediction Using Machine Learning

This project predicts the **final team score** in a T20/ODI match using current match stats such as boundaries, overs, and run rate.

> ✅ Built with Streamlit | ✅ Trained with 4 ML models | ✅ Validated using command-line inputs (CLI testing)

---

## 🔧 Technologies Used

- Python
- Pandas, NumPy
- Scikit-learn (Linear Regression, Decision Tree, Random Forest, KNN)
- Streamlit (UI)
- Matplotlib

---

## 📊 Features

- Predict total score based on:
  - Boundary Count (4s + 6s)
  - Overs Completed
  - Current Run Rate
- Supports **live prediction** via Streamlit for:
  - ✅ Decision Tree
  - ✅ KNN (with proper scaling)
- Compares all 4 models using evaluation metrics:
  - MAE, MSE, R² Score
- Includes visualizations:
  - Actual vs Predicted chart
  - R² bar chart

---

## 📓 Google Colab Notebook

The original development, data exploration, and model building were done in Google Colab.

▶️ [Open in Google Colab](https://colab.research.google.com/github/kasinadhan-in/cricket-score-prediction/blob/main/notebooks/cricket_score_prediction.ipynb)

This includes:
- Data cleaning and preprocessing
- Boundary/Overs/Run rate analysis
- Model experiments (LR, DT, RF, KNN)
- Visualizations & evaluation metrics



## 🧠 Model Evaluation Summary

| Model              | MAE    | MSE     | R² Score |
|-------------------|--------|---------|----------|
| Linear Regression | 13.25  | 385.88  | 0.926    |
| Decision Tree     | 12.95  | 307.55  | 0.941    |
| KNN Regressor     | 12.67  | 238.63  | 0.954    |
| Random Forest     | 9.89   | 155.87  | 0.970    |

---

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/kasinadhan-in/cricket-score-prediction.git
cd cricket-score-prediction
