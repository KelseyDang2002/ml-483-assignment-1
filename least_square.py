import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error as mse

TEST_DATA_PERCENTAGE = 0.3 # 30% testing and 70% training
POLYNOMIAL_ORDER = 1

'''Main'''
def main():
    X_train, X_test, y_train, y_test = split_data()
    linear_regression_model(X_train, X_test, y_train, y_test)

'''Split into testing and trainnig data'''
def split_data():
    # read csv file into dataframe
    df = pd.read_csv('wine_data.csv', sep=';')
    print(f"Dataframe:\n{df}\n")

    # drop label column to only get the features
    x = df.drop(['quality'], axis=1).values
    print(f"Training Features:\n{x}\n")

    # label column
    y = df['quality'].values
    print(f"Labels (quality):\n{y}\n")

    # split data
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=TEST_DATA_PERCENTAGE, random_state=0)
    return X_train, X_test, y_train, y_test

'''Plot'''
def plot(x_axis, y_axis, title):
    plt.scatter(x_axis, y_axis)
    plt.xlabel("Actual Quality")
    plt.ylabel("Predicted Quality")
    plt.title(title)
    plt.show()

'''Linear Regression'''
def linear_regression_model(X_train, X_test, y_train, y_test):
    # linear regression model
    lr = LinearRegression()

    # preprocess training data TODO
    poly = PolynomialFeatures(degree=POLYNOMIAL_ORDER, interaction_only=False)
    poly.fit_transform(X_train)

    # train the model with X_train and y_train
    lr.fit(X_train, y_train)
    
    # print y-intercept (b in mx + b)
    print(f"y-intercept: {lr.intercept_}\n")
    # print coefficients (m in mx + b), there should be 11 coefficients for the 11 training features
    print(f"11 Coefficients:\n{lr.coef_}\n")

    # predict quality based on X_test
    y_pred_test = lr.predict(X_test)
    plot(y_test, y_pred_test, "Testing")
    print(f"Testing R-Squared Score (close to 1 = better):\t{r2_score(y_test, y_pred_test)}")
    print(f"Testing RMSE Score (close to 0 = better):\t{np.sqrt(mse(y_test, y_pred_test))}\n")
    
    # predict quality based on X_train
    y_pred_train = lr.predict(X_train)
    plot(y_train, y_pred_train, "Training")
    print(f"Training R-Squared Score (close to 1 = better):\t{r2_score(y_train, y_pred_train)}")
    print(f"Training RMSE Score (close to 0 = better):\t{np.sqrt(mse(y_train, y_pred_train))}\n")

'''Call main'''
if __name__ == "__main__":
    main()
