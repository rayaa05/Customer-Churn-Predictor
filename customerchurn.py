import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("Telco-Customer-Churn.csv")

# ---------------------------
# BASIC CHECKS
# ---------------------------
print(df.head())
print(df.info())
print(df.isnull().sum())

# ---------------------------
# DROP UNNECESSARY COLUMN
# ---------------------------
df.drop("customerID", axis=1, inplace=True)

# ---------------------------
# HANDLE DATA TYPES
# ---------------------------
# Convert TotalCharges to numeric
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# ---------------------------
# HANDLE MISSING VALUES
# ---------------------------
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

# ---------------------------
# ENCODE TARGET VARIABLE
# ---------------------------
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# ---------------------------
# ENCODE CATEGORICAL VARIABLES
# ---------------------------
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

# ---------------------------
# FEATURE SCALING
# ---------------------------
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
df[num_cols] = scaler.fit_transform(df[num_cols])


# ---------------------------
# FINAL OUTPUT CHECK
# ---------------------------
print("\nPreprocessing Completed Successfully!\n")
print(df.head())

# ---------------------------
# MODEL STARTS HERE
# ---------------------------

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
