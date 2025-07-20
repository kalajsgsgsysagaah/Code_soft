# Credit Card Fraud Detection

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

from imblearn.over_sampling import SMOTE  # Install with: pip install imbalanced-learn

# Step 1: Load Dataset
df = pd.read_csv('creditcard.csv')  # Dataset link: https://www.kaggle.com/mlg-ulb/creditcardfraud
print("Shape:", df.shape)
print(df.head())

# Step 2: Check class distribution
print("\nClass Distribution:\n", df['Class'].value_counts())  # 0 = genuine, 1 = fraud

# Step 3: Normalize 'Amount' and drop 'Time'
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df.drop('Time', axis=1, inplace=True)

# Step 4: Define features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 6: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

print("\nAfter SMOTE Resampling:")
print(y_resampled.value_counts())

# Step 7: Train Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(X_resampled, y_resampled)

# Step 8: Predict on test data
y_pred = model.predict(X_test)

# Step 9: Evaluation
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Optional: Show key metrics separately
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"\nPrecision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

# Optional Visualization
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
