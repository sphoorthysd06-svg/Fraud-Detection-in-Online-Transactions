# MAJOR PROJECT - 2nd 
# FRAUD DETECTION IN ONLINE TRANSACTIONS
# Company: Fin-Care Starpowerz Digital Technology Pvt. Ltd.

# 1. IMPORT LIBRARIES
import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Read CSV file
df = pd.read_csv("Fraud_Detection_Cleaned.csv")

print("CSV Loaded:", df.shape)
print(df.head())

# NUMPY ANALYSIS

# Create Risk Level using NumPy
df["Risk_Level"] = np.where(
    df["Velocity_Score"] >= 60,
    "High",
    "Normal"
)

# Calculate average transaction amount
average_amount = np.mean(
    df["Transaction_Amount"]
)

# Find minimum and maximum transaction amount
minimum_amount = np.min(
    df["Transaction_Amount"]
)

maximum_amount = np.max(
    df["Transaction_Amount"]
)

# Calculate standard deviation
std_amount = np.std(
    df["Transaction_Amount"]
)

print("\n===== NUMPY ANALYSIS =====")

print(
    "Average Transaction Amount:",
    round(average_amount, 2)
)

print(
    "Minimum Transaction Amount:",
    minimum_amount
)

print(
    "Maximum Transaction Amount:",
    maximum_amount
)

print(
    "Standard Deviation:",
    round(std_amount, 2)
)

print("\nRisk Level Distribution:")

print(
    df["Risk_Level"].value_counts()
)

# 2. LOAD DATA FROM MYSQL
conn = pymysql.connect(
    host="localhost",
    user="root",
    password="system",
    database="fraud_detection_db"
)

df = pd.read_sql(
    "SELECT * FROM fraud_transactions",
    conn
)

conn.close()

print("Data Loaded:", df.shape)
print(df.head())

# 3. DATA CLEANING
df.columns = df.columns.str.strip()
df.drop_duplicates(inplace=True)

df["Transaction_Date"] = pd.to_datetime(
    df["Transaction_Date"],
    errors="coerce"
)

num_cols = [
    "Transaction_Amount",
    "Account_Age_Months",
    "Previous_Transactions",
    "Transactions_Last_24H",
    "Failed_Transactions_24H",
    "IP_Risk_Score",
    "Device_Risk_Score",
    "Velocity_Score"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col] = df[col].fillna(df[col].median())

print("\nMissing Values:")
print(df.isnull().sum().sum())

# 4. FEATURE ENGINEERING
df["Year"] = df["Transaction_Date"].dt.year
df["Month"] = df["Transaction_Date"].dt.month

# 5. FRAUD ANALYSIS

# Fraud Distribution
sns.countplot(x="Fraud_Status", data=df)
plt.title("Fraudulent vs Legitimate Transactions")
plt.show()

# Fraud by Transaction Type
sns.countplot(
    x="Transaction_Type",
    hue="Fraud_Status",
    data=df
)
plt.title("Fraud by Transaction Type")
plt.xticks(rotation=30)
plt.show()

# Risk Score Analysis
sns.boxplot(
    x="Fraud_Status",
    y="IP_Risk_Score",
    data=df
)
plt.title("IP Risk Score vs Fraud")
plt.show()

# Correlation
sns.heatmap(
    df[num_cols + ["Fraud_Flag"]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Fraud Risk Correlation")
plt.show()

# 6. PREPARE DATA FOR MACHINE LEARNING

# Select useful fraud-related features
features = [
    "Transaction_Amount",
    "Account_Age_Months",
    "Previous_Transactions",
    "Transactions_Last_24H",
    "Failed_Transactions_24H",
    "IP_Risk_Score",
    "Device_Risk_Score",
    "Velocity_Score"
]

X = df[features]
y = df["Fraud_Flag"]

# 7. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# RANDOM FOREST
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# 8.LINE GRAPH - ACTUAL VS PREDICTED

comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
}).reset_index(drop=True)

comparison.head(50).plot(
    kind="line",
    figsize=(10, 5)
)

plt.title("Actual vs Predicted Fraud")
plt.xlabel("Transaction")
plt.ylabel("Fraud Flag")
plt.legend(["Actual", "Predicted"])
plt.grid(True)
plt.show()

# 9.MODEL EVALUATION
print("\n===== MODEL RESULTS =====")

print(
    "Accuracy:",
    accuracy_score(y_test, y_pred)
)

print("\nClassification Report:")
print(
    classification_report(y_test, y_pred)
)

# 10.CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Fraud Detection Confusion Matrix")
plt.show()

# 11.FEATURE IMPORTANCE
importance = pd.Series(
    model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

importance.plot(kind="barh")

plt.title("Fraud Risk Factor Importance")
plt.xlabel("Importance")
plt.show()

# 12.PIE CHART - FRAUD DISTRIBUTION

fraud = df["Fraud_Status"].value_counts()

plt.figure(figsize=(6, 6))

plt.pie(
    fraud,
    labels=fraud.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Fraudulent vs Legitimate Transactions")
plt.show()

# 13. SAVE CLEANED DATA
df.to_csv(
    "Fraud_Detection_Cleaned.csv",
    index=False
)

print("\nFraud Detection Project Completed!")