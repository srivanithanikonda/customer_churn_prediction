import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Loaded Successfully")
print("Dataset Shape:", df.shape)

print(df.head())

print(df.columns.tolist())

# Check missing values
print(df.isnull().sum())

# Convert TotalCharges column
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

# Remove null values
df.dropna(inplace=True)

print(df.dtypes)

print(df.describe())

# Churn count
print(df["Churn"].value_counts())

# Create graphs folder
os.makedirs("graphs", exist_ok=True)

# Graph 1 - Churn Distribution
plt.figure(figsize=(6, 4))

sns.countplot(
    x="Churn",
    data=df
)

plt.title("Customer Churn Distribution")

plt.tight_layout()

plt.savefig("graphs/churn_distribution.png")

plt.close()

# Graph 2 - Contract vs Churn
plt.figure(figsize=(7, 4))

sns.countplot(
    x="Contract",
    hue="Churn",
    data=df
)

plt.title("Contract Type vs Churn")

plt.xticks(rotation=10)

plt.tight_layout()

plt.savefig("graphs/contract_vs_churn.png")

plt.close()

# Graph 3 - Monthly Charges vs Churn
plt.figure(figsize=(6, 4))

sns.boxplot(
    x="Churn",
    y="MonthlyCharges",
    data=df
)

plt.title("Monthly Charges vs Churn")

plt.tight_layout()

plt.savefig("graphs/monthly_charges_vs_churn.png")

plt.close()

# Graph 4 - Internet Service vs Churn
plt.figure(figsize=(7, 4))

sns.countplot(
    x="InternetService",
    hue="Churn",
    data=df
)

plt.title("Internet Service vs Churn")

plt.tight_layout()

plt.savefig("graphs/internet_service_vs_churn.png")

plt.close()

# Graph 5 - Tenure Distribution
plt.figure(figsize=(7, 4))

sns.histplot(
    data=df,
    x="tenure",
    hue="Churn",
    bins=30,
    kde=True
)

plt.title("Tenure Distribution by Churn")

plt.tight_layout()

plt.savefig("graphs/tenure_distribution.png")

plt.close()

print("Graphs Saved Successfully")

# Business insights
print("Business Insights")

print("1. Month-to-month customers have higher churn.")
print("2. Customers with high monthly charges are more likely to leave.")
print("3. Long-term customers show better retention.")
print("4. Fiber optic users have higher churn.")

# Data preprocessing
df.drop("customerID", axis=1, inplace=True)

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})

df = pd.get_dummies(
    df,
    drop_first=True
)

print(df.head())

# Features and target
X = df.drop("Churn", axis=1)

y = df["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Shape:", X_train.shape)

print("Testing Shape:", X_test.shape)

# Models
lr_model = LogisticRegression(max_iter=1000)

dt_model = DecisionTreeClassifier(
    random_state=42
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

models = {
    "Logistic Regression": lr_model,
    "Decision Tree": dt_model,
    "Random Forest": rf_model
}

results = {}

# Model training
for model_name, model in models.items():

    print(model_name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[model_name] = accuracy

    print("Accuracy Score:", round(accuracy * 100, 2), "%")

    print(confusion_matrix(y_test, y_pred))

    print(classification_report(y_test, y_pred))

# Model comparison table
results_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": list(results.values())
})

print(results_df)

# Accuracy graph
plt.figure(figsize=(7, 4))

model_names = list(results.keys())

accuracy_values = [
    value * 100 for value in results.values()
]

plt.bar(
    model_names,
    accuracy_values
)

plt.ylabel("Accuracy Percentage")

plt.title("Model Accuracy Comparison")

plt.ylim(70, 100)

for i, value in enumerate(accuracy_values):

    plt.text(
        i,
        value + 0.3,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()

plt.savefig("graphs/model_accuracy_comparison.png")

plt.close()

# Feature importance
feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
)

top_features = feature_importance.nlargest(10)

plt.figure(figsize=(8, 5))

top_features.sort_values(ascending=True).plot(
    kind="barh",
    figsize=(8,6)
)

plt.title("Top 10 Important Features")

plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.tight_layout()

plt.savefig("graphs/feature_importance.png",dpi=300)

plt.close()

print("Feature Importance Graph Saved sucessfully")

# Final conclusion
best_model = max(results, key=results.get)

print(f"{best_model} gave the best performance.")

print("Customer Churn Analysis Completed Successfully")
print("Graphs saved in the 'graphs' folder for visualizations.")