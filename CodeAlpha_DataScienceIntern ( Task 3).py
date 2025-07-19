# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 22:02:42 2025

@author: Haji
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv('C:/Users/Haji/Downloads/Advertising.csv')

# Drop the unnamed index column
df.drop(columns=['Unnamed: 0'], inplace=True)

# Exploratory Data Analysis
sns.pairplot(df, diag_kind='kde')
plt.suptitle("Ad Spend vs Sales", y=1.02)
plt.show()

# Correlation Heatmap
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# Split into features and target
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R² Score: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")

# Coefficients (Impact of ad channels)
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\nAdvertising Impact on Sales:")
print(coefficients)

# Plot actual vs predicted sales
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.grid(True)
plt.show()
