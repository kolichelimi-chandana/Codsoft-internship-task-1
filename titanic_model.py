# titanic_model.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("📂 Loading dataset...")
df = pd.read_csv("Titanic-Dataset.csv")
print("✅ Dataset loaded successfully!\n")

print("🧹 Cleaning and preprocessing data...")
df.dropna(subset=["Embarked"], inplace=True)
df["Age"].fillna(df["Age"].mean(), inplace=True)

# Encode categorical columns
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
df["Embarked"] = le.fit_transform(df["Embarked"])

# Define features and target
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
y = df["Survived"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("🤖 Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\n🎯 Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Feature importance chart
print("\n📊 Plotting feature importance...")
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
sns.barplot(x=feature_importance, y=feature_importance.index)
plt.title("Feature Importance in Titanic Survival Prediction")
plt.show()
