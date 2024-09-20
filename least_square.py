import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# read csv file into dataframe
df = pd.read_csv('wine_data.csv', sep=';')

# print dataframe
print(f"Dataframe:\n{df}\n")

# check for null values
print(f"Check for null values:\n{df.isnull().sum()}\n")

# drop lablel column to get training features
x = df.drop(columns='quality')
print(f"Training Features:\n{x}\n")

# labels
y = df['quality']
print(f"Labels:\n{y}\n")

# split data into 50% testing and 50% training
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=0)

# linear regression model
lr = LinearRegression()

# fit the training data to linear regression model
lr.fit(x_train, y_train)
# print y-intercept (b in mx + b)
print(f"y-intercept: {lr.intercept_}\n")
# print coefficients (m in mx + b), there should be 11 coefficients for the 11 training features
print(f"11 Coefficients:\n{lr.coef_}\n")

# training is done, now we test by predicting quality based on x_train
y_pred_train = lr.predict(x_train)

# print the array of predictions
print(f"Predicted Training Values:\n{y_pred_train}\n")

# now we plot to visualize how good the predictions are
# compare actual quality (y_train) vs predicted (y_pred_train)
plt.scatter(y_train, y_pred_train)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.show()

# measure prediction with R Squared
print(f"R-Squared Score: {r2_score(y_train, y_pred_train)}\n")

# predict quality based on x_test
y_pred_test = lr.predict(x_test)
print(f"Predicted Testing Values:\n{y_pred_test}\n")

plt.scatter(y_test, y_pred_test)
plt.xlabel("Actual Quality")
plt.ylabel("Predicted Quality")
plt.show()

# measure prediction with R Squared
print(f"R-Squared Score: {r2_score(y_test, y_pred_test)}\n")

# df.columns = [
#     'fixed acidity', 
#     'volatile acidity',
#     'citric acid',
#     'residual sugar',
#     'chlorides',
#     'free sulfur dioxide',
#     'total sulfur dioxide',
#     'density',
#     'pH',
#     'sulphates',
#     'alcohol',
#     'quality'
# ]
