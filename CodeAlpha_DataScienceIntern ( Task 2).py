# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 21:37:48 2025

@author: Haji
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
df = pd.read_csv('C:/Users/Haji/Downloads/car data.csv')

# Feature Engineering
df['Car_Age'] = 2025 - df['Year']  # Assuming current year is 2025
df.drop(['Car_Name', 'Year'], axis=1, inplace=True)  # Dropping unnecessary features

# Encode categorical variables
categorical_cols = ['Fuel_Type', 'Selling_type', 'Transmission']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split features and target
X = df.drop('Selling_Price', axis=1)
y = df['Selling_Price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

# Feature importance
feat_importance = pd.Series(model.feature_importances_, index=X.columns)
feat_importance.sort_values().plot(kind='barh', figsize=(10,6), title='Feature Importance')
plt.tight_layout()
plt.show()
