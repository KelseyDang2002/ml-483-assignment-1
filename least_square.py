import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def main():
    # read csv file into dataframe
    df = pd.read_csv('wine_data.csv', sep=';')

    # print dataframe
    print(f"Dataframe:\n{df}\n")

    # check for null values
    # print(f"Check for null values:\n{df.isnull().sum()}\n")

    # drop lablel column to only get the features
    features = df.drop(columns='quality')
    print(f"Training Features:\n{features}\n")

    # label column
    labels = df['quality']
    print(f"Labels:\n{labels}\n")

    # split data into 50% testing and 50% training
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=0)

    # preprocess data
    scalar = preprocessing.StandardScaler().fit(X_train)

    # linear regression model
    lr = LinearRegression()

    # train the model with X_train and y_train
    lr.fit(X_train, y_train)
    
    # print y-intercept (b in mx + b)
    print(f"y-intercept: {lr.intercept_}\n")
    
    # print coefficients (m in mx + b), there should be 11 coefficients for the 11 training features
    print(f"11 Coefficients:\n{lr.coef_}\n")
    
    # training is done, now we test by predicting quality based on X_train
    y_pred_train = lr.predict(X_train)
    print(f"Predicted Training Values:\n{y_pred_train}\n")

    # now we plot to visualize how good the predictions are
    # compare actual quality (y_train) vs predicted (y_pred_train)
    # plt.scatter(y_train, y_pred_train)
    # plt.xlabel("Actual Quality")
    # plt.ylabel("Predicted Quality")
    # plt.show()

    # measure prediction with R Squared
    print(f"Training R-Squared Score: {r2_score(y_train, y_pred_train)}\n")

    # predict quality based on X_test
    y_pred_test = lr.predict(X_test)
    print(f"Predicted Testing Values:\n{y_pred_test}\n")

    # plt.scatter(y_test, y_pred_test)
    # plt.xlabel("Actual Quality")
    # plt.ylabel("Predicted Quality")
    # plt.show()

    # measure prediction with R Squared
    print(f"Testing R-Squared Score: {r2_score(y_test, y_pred_test)}\n")

if __name__ == "__main__":
    main()
