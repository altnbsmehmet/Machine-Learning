#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the dataset
dataset = pd.read_csv('Multiple_Linear_Regression/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X)) 

#splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#training the multiple linear regression model on the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)
np.set_printoptions(precision = 2)
print(f"Predicted results: \n {np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1 )} \n")

#visualizing the comparison of the test set and the predicted set results
indexes = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
plt.plot(indexes, y_pred, color = 'red', label='Real Values')
plt.plot(indexes, y_test, color = 'blue', label='Predicted Values')
plt.xlabel('Index')
plt.ylabel('Values')
plt.title('Comparison of Real and Predicted Values')
plt.legend()
plt.show()

#making a single prediction
print(f"Single prediction depending on State = California,R&D Spend = 160000, Administration Spend = 130000, Marketing Spend = 300000: {regressor.predict([[1, 0, 0, 160000, 130000, 300000]])} \n")

#final regression equation with the final values of the coefficient
print(f"Coefficients: {regressor.coef_} \n")
print(f"Intercept: {regressor.intercept_} \n")

#evaluating the model with r-squared value
from sklearn.metrics import r2_score
print(f"R-Squared value: {r2_score(y_test, y_pred)}")