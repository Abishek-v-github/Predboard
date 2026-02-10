import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score
)


st.set_page_config(page_title="PredictBoard", layout="wide")
st.title("📈 PredictBoard")


if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
    st.session_state.is_regression = None
    st.session_state.x_cols = None
    st.session_state.y_col = None
    st.session_state.df = None


SAMPLE_DATASETS = {
    "House Prices (Regression)": """
area_sqft,price
800,40000
1000,55000
1200,65000
1500,80000
1800,95000
2200,120000
""",
    "Salary vs Experience (Regression)": """
experience_years,salary
0,25000
1,30000
2,35000
3,42000
4,50000
5,60000
6,72000
""",
    "Student Pass/Fail (Classification)": """
study_hours,attendance,passed
1,60,0
2,65,0
3,70,0
4,75,1
5,80,1
6,85,1
"""
}

def load_sample_csv(text):
    return pd.read_csv(StringIO(text.strip()))


st.subheader("📂 Data Source")

source = st.radio(
    "Choose dataset source:",
    ["Use sample dataset", "Upload CSV"]
)

df = None

if source == "Use sample dataset":
    name = st.selectbox("Select sample dataset", list(SAMPLE_DATASETS.keys()))
    df = load_sample_csv(SAMPLE_DATASETS[name])
else:
    file = st.file_uploader("Upload a CSV", type=["csv"])
    if file:
        df = pd.read_csv(file)

if df is None:
    st.info("Please select or upload a dataset.")
    st.stop()

st.session_state.df = df.copy()

st.subheader("🔍 Dataset Preview")
st.dataframe(df.head())


st.sidebar.header("⚙️ Data Preprocessing")

null_strategy = st.sidebar.selectbox(
    "Missing Value Strategy",
    ["None", "Drop Rows", "Fill with Mean", "Fill with Median"]
)

if null_strategy == "Drop Rows":
    df = df.dropna()
elif null_strategy == "Fill with Mean":
    df = df.fillna(df.mean(numeric_only=True))
elif null_strategy == "Fill with Median":
    df = df.fillna(df.median(numeric_only=True))

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
if cat_cols:
    encode_cols = st.sidebar.multiselect("Encode Categorical Columns", cat_cols)
    for col in encode_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

st.subheader("🎯 Task & Feature Selection")

problem_type = st.radio("Choose problem type", ["Regression", "Classification"])
is_regression = problem_type == "Regression"

if is_regression:
    x_cols = [st.selectbox("Select Feature (X)", df.columns)]
    y_col = st.selectbox("Select Target (Y)", df.columns)
else:
    x_cols = st.multiselect("Select Features (X)", df.columns)
    y_col = st.selectbox("Select Target (Y)", df.columns)

X = df[x_cols]
y = df[y_col]


st.sidebar.header("🤖 Model Selection")

if is_regression:
    model_name = st.sidebar.selectbox(
        "Choose Regressor",
        ["Linear Regression", "Decision Tree"]
    )
    if model_name == "Linear Regression":
        model = LinearRegression()
    else:
        depth = st.sidebar.slider("Max Depth", 1, 20, 5)
        model = DecisionTreeRegressor(max_depth=depth, random_state=42)

else:
    model_name = st.sidebar.selectbox(
        "Choose Classifier",
        ["Logistic Regression", "KNN"]
    )
    if model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    else:
        k = st.sidebar.slider("K for KNN", 1, 15, 5)
        model = KNeighborsClassifier(n_neighbors=k)


test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20) / 100
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42
)


if st.button("🚀 Train & Evaluate Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.session_state.trained_model = model
    st.session_state.is_regression = is_regression
    st.session_state.x_cols = x_cols
    st.session_state.y_col = y_col

    st.subheader("📊 Model Performance")

    if is_regression:
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mae:.2f}")
        c2.metric("RMSE", f"{rmse:.2f}")
        c3.metric("R²", f"{r2:.2f}")

        st.subheader("📈 Actual vs Predicted")

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_test, y_pred, alpha=0.7)
        min_v = min(y_test.min(), y_pred.min())
        max_v = max(y_test.max(), y_pred.max())
        ax.plot([min_v, max_v], [min_v, max_v], "r--")

        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{model_name} – Actual vs Predicted")
        ax.grid(True)
        st.pyplot(fig)

    else:
        acc = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{acc:.2f}")

# ===============================
# Prediction Section
# ===============================
st.subheader("🔮 Make a Prediction")

if st.session_state.trained_model is None:
    st.info("Train a model to enable predictions.")
    st.stop()

model = st.session_state.trained_model

if st.session_state.is_regression:
    val = st.number_input(
        f"Enter value for {st.session_state.x_cols[0]}",
        key="reg_input"
    )
    if st.button("Predict"):
        pred = model.predict([[val]])[0]
        st.success(f"Predicted {st.session_state.y_col}: {pred:.2f}")

else:
    inputs = []
    for col in st.session_state.x_cols:
        v = st.number_input(col, key=f"cls_{col}")
        inputs.append(v)

    if st.button("Predict"):
        pred = model.predict([inputs])[0]
        st.success(f"Predicted {st.session_state.y_col}: {pred}")
