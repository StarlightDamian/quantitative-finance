# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:14:03 2023

@author: awei
"""

from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Generate some sample data
np.random.seed(42)
X = np.random.rand(100, 5)  # 100 samples, 5 features
y1 = 2 * X[:, 0] + 3 * X[:, 1] + np.random.normal(0, 0.1, 100)  # First target
y2 = -1 * X[:, 2] + 0.5 * X[:, 3] + np.random.normal(0, 0.1, 100)  # Second target
y = np.column_stack((y1, y2))  # Stack the targets horizontally

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
base_regressor = LinearRegression()

# Create a MultiOutputRegressor with the base model
multioutput_regressor = MultiOutputRegressor(base_regressor)

# Fit the model to the training data
multioutput_regressor.fit(X_train, y_train)

# Make predictions on the test data
y_pred = multioutput_regressor.predict(X_test)

# Evaluate the performance
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
