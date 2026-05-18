import pandas as pd
df = pd.read_csv("WA_Fn-UseC_-telco-Customer-Churn.csv")
print(df.head())
print(df.shape)
print(df.columns)
print(df["Churn"].value_counts())
print(df.isnull().sum())
print(df.dtypes)
print(df["TotalCharges"].head())
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
print(df["TotalCharges"].dtype)
print(df.describe())
df.dropna(subset=["TotalCharges"], inplace=True)
df.drop(columns=["customerID"], inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
print(df.dtypes)
df = pd.get_dummies(df, drop_first=True)
print(df.head())
print(df.shape)
X = df.drop("Churn", axis=1)
y = df["Churn"]
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model trained sucessfully")
from sklearn.metrics import accuracy_score 
y_pred = model.predict(X_test)

accuracy  = accuracy_score(y_test, y_pred)  
print("Accuracy:", accuracy)  
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

cm = confusion_matrix(y_test, y_pred)

print(cm)

print(classification_report(y_test, y_pred))
