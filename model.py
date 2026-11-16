import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(page_title="Customer Churn App", layout="wide")

# ---------------------------
# LOAD DATA
# ---------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    return df

df = load_data()

# ---------------------------
# PREPROCESSING
# ---------------------------
df.drop("customerID", axis=1, inplace=True)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# Save original for UI
original_df = df.copy()

df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

le_dict = {}
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

scaler = StandardScaler()
df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.fit_transform(
    df[["tenure", "MonthlyCharges", "TotalCharges"]]
)

# ---------------------------
# MODEL
# ---------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

# ---------------------------
# SIDEBAR NAVIGATION
# ---------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Dashboard", "Prediction"])

# ---------------------------
# DASHBOARD PAGE
# ---------------------------
if page == "Dashboard":

    st.title("📊 Customer Churn Analytics Dashboard")

    # KPI METRICS
    col1, col2, col3 = st.columns(3)

    churn_rate = original_df["Churn"].value_counts(normalize=True)["Yes"] * 100
    avg_monthly = original_df["MonthlyCharges"].mean()
    avg_tenure = original_df["tenure"].mean()

    col1.metric("Churn Rate", f"{churn_rate:.2f}%")
    col2.metric("Avg Monthly Charges", f"{avg_monthly:.2f}")
    col3.metric("Avg Tenure", f"{avg_tenure:.1f} months")

    st.markdown("---")

    # FILTERS
    st.subheader("🎛️ Filter Data")

    tenure_range = st.slider("Tenure", 0, 72, (0, 72))
    monthly_range = st.slider("Monthly Charges", 0.0, 200.0, (0.0, 200.0))
    contract_filter = st.multiselect(
        "Contract Type",
        options=original_df["Contract"].unique(),
        default=original_df["Contract"].unique()
    )

    filtered_df = original_df[
        (original_df["tenure"] >= tenure_range[0]) &
        (original_df["tenure"] <= tenure_range[1]) &
        (original_df["MonthlyCharges"] >= monthly_range[0]) &
        (original_df["MonthlyCharges"] <= monthly_range[1]) &
        (original_df["Contract"].isin(contract_filter))
    ]

    st.write(f"Showing {len(filtered_df)} records")

    st.markdown("---")

    # VISUALS
    col1, col2 = st.columns(2)

    with col1:
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Churn', data=filtered_df, ax=ax1)
        ax1.set_title("Churn Distribution")
        st.pyplot(fig1)

    with col2:
        fig2, ax2 = plt.subplots()
        sns.histplot(filtered_df['MonthlyCharges'], kde=True, ax=ax2)
        ax2.set_title("Monthly Charges")
        st.pyplot(fig2)

    col3, col4 = st.columns(2)

    with col3:
        fig3, ax3 = plt.subplots()
        sns.boxplot(x=filtered_df['TotalCharges'], ax=ax3)
        ax3.set_title("Total Charges")
        st.pyplot(fig3)

    with col4:
        fig4, ax4 = plt.subplots()
        sns.countplot(x='Contract', hue='Churn', data=filtered_df, ax=ax4)
        ax4.set_title("Contract vs Churn")
        st.pyplot(fig4)

# ---------------------------
# PREDICTION PAGE
# ---------------------------
elif page == "Prediction":

    st.title("🔮 Customer Churn Prediction")

    st.subheader("Enter Customer Details")

    gender = st.selectbox("Gender", le_dict["gender"].classes_)
    contract = st.selectbox("Contract", le_dict["Contract"].classes_)
    tenure = st.slider("Tenure", 0, 72, 12)
    monthly = st.slider("Monthly Charges", 0.0, 200.0, 50.0)
    total = st.slider("Total Charges", 0.0, 10000.0, 1000.0)

    if st.button("Predict"):

        input_dict = {col: 0 for col in X.columns}

        input_dict["gender"] = le_dict["gender"].transform([gender])[0]
        input_dict["Contract"] = le_dict["Contract"].transform([contract])[0]
        input_dict["tenure"] = tenure
        input_dict["MonthlyCharges"] = monthly
        input_dict["TotalCharges"] = total

        input_df = pd.DataFrame([input_dict])

        # Scale numeric
        input_df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
            input_df[["tenure", "MonthlyCharges", "TotalCharges"]]
        )

        prediction = model.predict(input_df)

        if prediction[0] == 1:
            st.error("⚠️ Customer is likely to CHURN")
        else:
            st.success("✅ Customer will STAY")

    st.markdown("---")
    st.subheader("Model Accuracy")
    st.write(f"{accuracy:.2f}")