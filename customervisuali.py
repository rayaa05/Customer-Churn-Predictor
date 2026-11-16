import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------
# DATA PREPROCESSING
# ---------------------------
df = pd.read_csv("Telco-Customer-Churn.csv")
df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.fit_transform(
    df[["tenure", "MonthlyCharges", "TotalCharges"]]
)

print("Preprocessing Done!\n")

# ---------------------------
# VISUALIZATION
# ---------------------------
sns.set_style("whitegrid")
sns.set_palette("pastel")

# 1. Churn Distribution
plt.figure()
ax = sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution (Bar Chart)\nShows proportion of customers who stayed vs left")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
total = len(df)
for p in ax.patches:
    percentage = f'{100 * p.get_height()/total:.1f}%'
    ax.annotate(percentage, (p.get_x()+0.2, p.get_height()+50))
plt.show()

# 2. Churn by Contract
plt.figure()
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract Type\nCompares churn across contracts")
plt.show()

# 3. Distributions
fig, axes = plt.subplots(1, 2, figsize=(10,4))
sns.histplot(df['tenure'], kde=True, ax=axes[0])
axes[0].set_title("Tenure Distribution")
sns.histplot(df['MonthlyCharges'], kde=True, ax=axes[1])
axes[1].set_title("Monthly Charges Distribution")
plt.tight_layout()
plt.show()

# 4. Boxplot
plt.figure()
sns.boxplot(x=df['TotalCharges'])
plt.title("Outlier Detection (Boxplot)")
plt.show()

# 5. Heatmap
plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True)
plt.title("Correlation Heatmap")
plt.show()

# 6. Pie Chart
plt.figure()
df['Churn'].value_counts().plot.pie(autopct='%1.1f%%')
plt.title("Churn Percentage (Pie Chart)")
plt.ylabel("")
plt.show()

# 7. Pairplot
sns.pairplot(df[['tenure','MonthlyCharges','TotalCharges','Churn']], hue='Churn')
plt.suptitle("Pairplot of Key Features", y=1.02)
plt.show()


# ---------------------------
# DASHBOARD
# ---------------------------
fig, axes = plt.subplots(2, 2, figsize=(12,8))
sns.countplot(x='Churn', data=df, ax=axes[0,0])
axes[0,0].set_title("Churn Distribution")
sns.countplot(x='Contract', hue='Churn', data=df, ax=axes[0,1])
axes[0,1].set_title("Contract vs Churn")
sns.histplot(df['MonthlyCharges'], kde=True, ax=axes[1,0])
axes[1,0].set_title("Monthly Charges")
sns.boxplot(x=df['TotalCharges'], ax=axes[1,1])
axes[1,1].set_title("Total Charges Outliers")
plt.suptitle("Customer Churn Dashboard", fontsize=14)
plt.tight_layout()
plt.show()


# ---------------------------
# FINAL TEXT CONCLUSION
# ---------------------------
print("\nFINAL CONCLUSION:\n")
print("The analysis shows that customer churn is influenced by multiple factors such as contract type, monthly charges, and tenure.")
print("Customers with higher monthly charges and shorter tenure are more likely to churn.")
print("Additionally, contract type plays a significant role, where flexible or short-term contracts show higher churn rates.")
print("Through visualization, we were able to identify patterns, detect outliers, and understand relationships between variables.")
print("This analysis can help telecom companies take proactive steps to improve customer retention and reduce churn.\n")
