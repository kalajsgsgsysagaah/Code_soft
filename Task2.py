# Movie Rating Prediction - Full Code
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv('IMDB-Movie-Data.csv')  # Rename if different
print("Initial shape:", df.shape)

# Basic inspection
print(df.columns)
print(df[['Title', 'Genre', 'Director', 'Actors', 'Rating']].head())

# Drop unnecessary columns
df.drop(['Title', 'Description', 'Votes', 'Revenue (Millions)', 'Metascore'], axis=1, inplace=True, errors='ignore')

# Drop rows with missing values
df.dropna(inplace=True)

# Optional: take only top genres, directors, actors
top_genres = df['Genre'].value_counts().index[:5]
df = df[df['Genre'].isin(top_genres)]

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['Genre', 'Director'], drop_first=True)

# Simplify 'Actors' by using only the first actor listed
df['Actor_1'] = df['Actors'].apply(lambda x: x.split(',')[0])
df.drop('Actors', axis=1, inplace=True)

df = pd.get_dummies(df, columns=['Actor_1'], drop_first=True)

# Features and target
X = df.drop('Rating', axis=1)
y = df['Rating']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("\nMean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Plot actual vs predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Movie Ratings")
plt.show()
