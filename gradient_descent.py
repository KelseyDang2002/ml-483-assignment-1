import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import time

TEST_DATA_PERCENTAGE = 0.2
POLYNOMIAL_ORDER = 3

'''Main'''
def main():
    x, y = features_and_label()
    X_train, X_test, y_train, y_test = split_data(x, y)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    order = 1
    learning_rate = 0.000000132
    iteration = 2000
    
    while order <= POLYNOMIAL_ORDER:
        try:
            theta, cost_list = gradient_descent_model(X_train, X_test, y_train, y_test, order, learning_rate, iteration)
            print(f"Theta ({len(theta)}): {theta}")
            order += 1
        
        except KeyboardInterrupt:
            print("\nKeyboard Interrupt")
            break

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
def gradient_descent_model(X_train, X_test, y_train, y_test, order, learning_rate, iteration):
    X_train_poly, X_test_poly = preprocess_dataset(X_train, X_test, order)
    print(f"Learning Rate: {learning_rate:.9f}")
    print(f"Iterations: {iteration}\n")
    
    # add column of ones to account for y-intercept
    X_train_poly = np.hstack((np.ones((X_train_poly.shape[0], 1)), X_train_poly))
    X_test_poly = np.hstack((np.ones((X_test_poly.shape[0], 1)), X_test_poly))

    m = y_train.size
    theta = np.zeros((X_train_poly.shape[1]))
    cost_list = []

    start = time.time()

    # Stochastic Gradient Descent loop
    for iteration in range(iteration):
        for i in range(m):
            # pick random sample from training data
            random_sample = np.random.randint(m)
            X_index = X_train_poly[random_sample:random_sample + 1] # select i-th example
            y_index = y_train[random_sample:random_sample + 1] # select i-th example

            # predict using current theta
            y_pred_index = np.dot(X_index, theta)

            gradient = X_index.T.dot(y_pred_index - y_index)
            theta = theta - learning_rate * gradient
        
        y_pred = np.dot(X_train_poly, theta)
        cost = (1/(2*m)) * np.sum(np.square(y_pred - y_train))
        cost_list.append(cost)
    
        # print cost at every 100th iteration
        if iteration % 100 == 0:
            print(f"Iteration: {iteration}, Cost: {cost:.6f}")

    end = time.time()

    train_rmse = np.sqrt(mean_squared_error(y_train, np.dot(X_train_poly, theta)))
    train_r2 = r2_score(y_train, np.dot(X_train_poly, theta))
    print(f"\nTraining RMSE Score (close to 0 = better):\t{train_rmse:.6f}")
    print(f"Training R-Squared Score (close to 1 = better):\t{train_r2:.6f}\n")

    test_rmse = np.sqrt(mean_squared_error(y_test, np.dot(X_test_poly, theta)))
    test_r2 = r2_score(y_test, np.dot(X_test_poly, theta))
    print(f"Testing RMSE Score (close to 0 = better):\t{test_rmse:.6f}")
    print(f"Testing R-Squared Score (close to 1 = better):\t{test_r2:.6f}\n")
        
    print(f"Training time: {end - start} seconds\n")

    return theta, cost_list

'''Call main'''
if __name__ == "__main__":
    main()
