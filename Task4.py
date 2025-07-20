# Sales Prediction Using Machine Learning

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load dataset
df = pd.read_csv('sales.csv')  # Ensure the file has columns like TV, Radio, Newspaper, Sales
print("First 5 rows of data:\n", df.head())

# Step 2: Explore and visualize
print("\nDataset Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

sns.pairplot(df)
plt.suptitle("Feature Relationships", y=1.02)
plt.show()

# Step 3: Feature and target selection
X = df[['TV', 'Radio', 'Newspaper']]  # Independent variables (ad spend)
y = df['Sales']                       # Dependent variable (target)

# Step 4: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Prediction
y_pred = model.predict(X_test)

# Step 7: Evaluation
print("\nModel Evaluation:")
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("Root Mean Squared Error (RMSE):", mean_squared_error(y_test, y_pred, squared=False))
print("R² Score:", r2_score(y_test, y_pred))

# Step 8: Visualize actual vs predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
