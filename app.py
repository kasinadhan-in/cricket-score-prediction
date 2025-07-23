import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ------------------------
# Load and Preprocess Data
# ------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("batting_summary.csv")
    df.fillna(0, inplace=True)
    df['Strike_Rate'] = pd.to_numeric(df['Strike_Rate'], errors='coerce')
    df['Boundary_Count'] = df['4s'] + df['6s']

    team_stats = df.groupby(['Match_no', 'Team_Innings']).agg({
        'Runs': 'sum',
        'Balls': 'sum',
        'Boundary_Count': 'sum',
        'Strike_Rate': 'mean'
    }).reset_index()

    team_stats['Overs'] = team_stats['Balls'] // 6 + (team_stats['Balls'] % 6) / 10
    team_stats['Run_Rate'] = team_stats['Runs'] / team_stats['Overs']
    return team_stats

data = load_data()
X = data[['Boundary_Count', 'Overs', 'Run_Rate']]
y = data['Runs']

# 🔁 Fit scaler on the full dataset (as done in CLI)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the scaled data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train models
lr_model = LinearRegression().fit(X_train, y_train)
dt_model = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
knn_model = KNeighborsRegressor(n_neighbors=5).fit(X_train, y_train)

# ------------------------
# Streamlit Interface
# ------------------------
st.title("🏏 Cricket Score Predictor")
st.markdown("Predict final team score based on match stats.\n\n✅ **Only Decision Tree and KNN** support interactive predictions (like CLI).")

# Sidebar Input
st.sidebar.header("Enter Match Stats (for DT & KNN only)")
boundary = st.sidebar.number_input("Boundary Count (4s + 6s)", min_value=0.0)
overs = st.sidebar.number_input("Overs Completed", min_value=0.0, max_value=20.0, step=0.1)
run_rate = st.sidebar.number_input("Current Run Rate", min_value=0.0, step=0.1)

model_choice = st.sidebar.selectbox("Select Model for Prediction", ['Decision Tree', 'KNN'])
input_data = np.array([[boundary, overs, run_rate]])
input_scaled = scaler.transform(input_data)

# Prediction
if st.sidebar.button("Predict Score"):
    if model_choice == 'KNN':
        pred = knn_model.predict(input_scaled)[0]
        y_pred = knn_model.predict(X_test)
    else:
        pred = dt_model.predict(input_scaled)[0]
        y_pred = dt_model.predict(X_test)

    st.subheader(f"🎯 Predicted Final Score: **{pred:.2f}** runs")

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.markdown("### 📊 Model Evaluation")
    st.write(f"- **MAE:** {mae:.2f}")
    st.write(f"- **MSE:** {mse:.2f}")
    st.write(f"- **R² Score:** {r2:.2f}")

    # Scatter Plot
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred, alpha=0.6, color='blue')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual Runs")
    ax.set_ylabel("Predicted Runs")
    ax.set_title(f"{model_choice}: Actual vs Predicted")
    st.pyplot(fig)

# ------------------------
# All Model Comparison Table
# ------------------------
st.markdown("## 📈 Model Comparison (All 4 Models)")
all_models = {
    'Linear Regression': lr_model,
    'Decision Tree': dt_model,
    'Random Forest': rf_model,
    'KNN': knn_model
}

model_metrics = {}
for name, model in all_models.items():
    preds = model.predict(X_test)
    model_metrics[name] = {
        'MAE': mean_absolute_error(y_test, preds),
        'MSE': mean_squared_error(y_test, preds),
        'R²': r2_score(y_test, preds)
    }

eval_df = pd.DataFrame(model_metrics).T.round(2)
st.dataframe(eval_df)

# R² Bar Chart
st.markdown("### 🔍 R² Score Chart")
st.bar_chart(eval_df[['R²']])


