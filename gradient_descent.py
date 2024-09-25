import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

TEST_DATA_PERCENTAGE = 0.2
POLYNOMIAL_ORDER = 1

'''Main'''
def main():
    x, y = features_and_label()
    X_train, X_test, y_train, y_test = split_data(x, y)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    order = 1
    theta = np.random.randn((X_train.shape[1], 1)) # error here
    learning_rate = 0.01
    iteration = 1000
    
    while order <= POLYNOMIAL_ORDER:
        try:
            theta, cost_list = gradient_descent_model(X_train, X_test, y_train, y_test, order, theta, learning_rate, iteration)
            order += 1
        
        except KeyboardInterrupt:
            print("Keyboard Interrupt\n")
            break

# '''Plot'''
# def plot(x_axis, y_axis, title, x_label, y_label):
#     plt.scatter(x_axis, y_axis)
#     plt.plot(x_axis, y_axis, c="red")
#     plt.title(title)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.xticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     plt.yticks([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#     plt.show()

'''Get training features and label column from data set'''
def features_and_label():
    # read csv file into dataframe
    df = pd.read_csv('wine_data.csv', sep=';')
        
    # check for any null or missing values
    print(f"Check for null or missing values:\n{df.isnull().sum()}\n")

    # get training features
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
    return x, y

'''Split into testing and trainnig data'''
def split_data(x, y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=TEST_DATA_PERCENTAGE, random_state=0)
    return X_train, X_test, y_train, y_test

'''Preprocess dataset'''
def preprocess_dataset(X_train, X_test, order):
    # preprocess data
    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.fit_transform(X_test)

    # polynomial features
    poly = PolynomialFeatures(degree=order, interaction_only=False, include_bias=False)
    X_train_poly = poly.fit_transform(X_train_scaled)
    X_test_poly = poly.fit_transform(X_test_scaled)

    print("-" * 75)
    print(f"Polynomial Order: {order}\n")

    return X_train_poly, X_test_poly

'''Gradient Descent model'''
def gradient_descent_model(X_train, X_test, y_train, y_test, order, theta, learning_rate, iteration):
    X_train_poly, X_test_poly = preprocess_dataset(X_train, X_test, order)
    
    start = time.time()

    m = y_train.size
    # theta = np.random.randn((X_train[1], 1)) # error here, define theta outside of function?
    cost_list = []

    for i in range(iteration):
        y_pred = np.dot(X_train, theta)
        theta = theta - (1/m) * learning_rate * (np.dot(X_train.T,(y_pred - y_train)))
        cost = (1/(2*m)) * np.sum(np.square(y_pred - y_train))
        cost_list.append(cost)
        
        if (i % (iteration / 10) == 0):
            print(f"Cost: {cost}")

    end = time.time()
    print(f"Training time: {end - start} seconds\n")
    return theta, cost_list

'''Call main'''
if __name__ == "__main__":
    main()
