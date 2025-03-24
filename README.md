# Movie Rating Analysis System

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

movies = pd.read_csv("movies.csv")

print("\n--- Movie Dataset Preview ---")
print(movies.head())

print("\n--- Dataset Info ---")
print(movies.info())

print("\n--- Null Values ---")
print(movies.isnull().sum())

movies = movies.dropna()

print("\n--- Basic Statistics ---")
print(movies.describe())

plt.figure(figsize=(8,5))
sns.histplot(movies['Rating'], bins=20, kde=True, color='orange')
plt.title("Distribution of Movie Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.grid()
plt.show()

plt.figure(figsize=(12,6))
genre_rating = movies.groupby('Genre')['Rating'].mean().sort_values(ascending=False)
genre_rating.plot(kind='bar', color='skyblue')
plt.title("Average Rating by Genre")
plt.ylabel("Average Rating")
plt.xticks(rotation=45)
plt.grid()
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(movies.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

features = movies[["Votes", "Year", "Runtime"]]
target = movies["Rating"]

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\n--- Model Evaluation ---")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

comparison = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print("\n--- Actual vs Predicted Ratings ---")
print(comparison.head())

import joblib
joblib.dump(model, "movie_rating_model.pkl")
print("Model saved as 'movie_rating_model.pkl'")
