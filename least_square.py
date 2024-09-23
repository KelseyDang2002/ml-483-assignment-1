import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error as mse
import time

TEST_DATA_PERCENTAGE = 0.3 # 30% testing and 70% training
POLYNOMIAL_ORDER = 5 # change order for testing

'''Main'''
def main():
    X_train, X_test, y_train, y_test = split_data()
    linear_regression_model(X_train, X_test, y_train, y_test)

'''Plot'''
def plot(x_axis, y_axis, title, x_label, y_label):
    plt.scatter(x_axis, y_axis)
    plt.plot(x_axis, y_axis, c="red")
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # plt.yticks([0, 1])
    plt.show()

# TODO
def features_and_label():
    return

'''Split into testing and trainnig data'''
def split_data():
    # read csv file into dataframe
    df = pd.read_csv('wine_data.csv', sep=';')
    # print(f"Dataframe:\n{df}\n")
        
    # check for any null or missing values
    print(f"Check for null or missing values:\n{df.isnull().sum()}\n")

    # drop label column to only get the features
    # x = df.drop(['quality'], axis=1).values
    x = df[["fixed acidity",
            "volatile acidity",
            "citric acid",
            "residual sugar",
            "chlorides",
            "free sulfur dioxide",
            "total sulfur dioxide",
            "density",
            "pH",
            "sulphates",
            "alcohol"
        ]].values
    print(f"Training Features:\n{x}\n")

    # label only column
    y = df['quality'].values
    print(f"Labels (quality):\n{y}\n")

    # split data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=TEST_DATA_PERCENTAGE, random_state=0)
    return X_train, X_test, y_train, y_test

'''Linear Regression'''
def linear_regression_model(X_train, X_test, y_train, y_test):
    # linear regression model
    lr = LinearRegression()

    # preprocess data
    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.fit_transform(X_test)

    # polynomial features
    poly = PolynomialFeatures(degree=POLYNOMIAL_ORDER, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.fit_transform(X_test_scaled)

    start = time.time()

    # train the model with X_train and y_train
    lr.fit(X_train_poly, y_train)
    
    end = time.time()
    print("-" * 75)
    print(f"Training time: {end - start} seconds\n")

    print(f"Testing Data: {TEST_DATA_PERCENTAGE * 100}%\nTraining Data: {100 - (TEST_DATA_PERCENTAGE * 100)}%\n")

    print(f"Polynomial Order: {POLYNOMIAL_ORDER}\n")
    
    # print y-intercept (b in mx + b)
    print(f"y-intercept: {lr.intercept_}")
    # print coefficients (m in mx + b)
    print(f"Coefficients ({len(lr.coef_)}):\n{lr.coef_}\n")

    # predict quality based on X_train
    y_pred_train = lr.predict(X_train_poly)
    # plot(y_train, y_pred_train, "Training", "Actual", "Predicted")
    print(f"Training RMSE Score (close to 0 = better):\t{np.sqrt(mse(y_train, y_pred_train)):.6f}")
    print(f"Training R-Squared Score (close to 1 = better):\t{r2_score(y_train, y_pred_train):.6f}\n")
    
    # predict quality based on X_test
    y_pred_test = lr.predict(X_test_poly)
    # plot(y_test, y_pred_test, "Testing", "Actual", "Predicted")
    print(f"Testing RMSE Score (close to 0 = better):\t{np.sqrt(mse(y_test, y_pred_test)):.6f}")
    print(f"Testing R-Squared Score (close to 1 = better):\t{r2_score(y_test, y_pred_test):.6f}\n")

# TODO
# def least_squares_solution(x, y):
#     X_T_X_inv = np.linalg.inv(x.T.dot(x))
#     w = X_T_X_inv.dot(x.T).dot(y)
#     return w

# Example: Using training data (for linear model) TODO
# X_train_poly = np.hstack([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled])  # Add intercept term
# w_manual = least_squares_solution(X_train_poly, y_train)

'''Call main'''
if __name__ == "__main__":
    main()
