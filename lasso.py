import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score, mean_squared_error
import time

TEST_DATA_PERCENTAGE = 0.2
POLYNOMIAL_ORDER = 5

'''Main'''
def main():
    x, y = features_and_label()
    X_train, X_test, y_train, y_test = split_data(x, y)
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    order = 1
    alpha = 0.00212
    
    while order <= POLYNOMIAL_ORDER:
        try:
            intercept, coefficients, X_train_selected, X_test_selected = lasso_model(X_train, X_test, y_train, y_test, order, alpha)
            print(f"Intercept: {intercept}")
            print(f"Coefficients ({len(coefficients)}): {coefficients}\n")
            intercept, coefficients, X_train_selected, X_test_selected = lasso_model(X_train_selected, X_test_selected, y_train, y_test, order, alpha)
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

'''Lasso model'''
def lasso_model(X_train, X_test, y_train, y_test, order, alpha):
    X_train_poly, X_test_poly = preprocess_dataset(X_train, X_test, order)
    print(f"Alpha value: {alpha}\n")

    # lasso regression model
    lasso = Lasso(alpha=alpha)

    start = time.time()

    # train the model
    lasso.fit(X_train_poly, y_train)

    end = time.time()
    print(f"Training time: {end - start} seconds\n")

    print(f"Testing Data: {TEST_DATA_PERCENTAGE * 100}%\nTraining Data: {100 - (TEST_DATA_PERCENTAGE * 100)}%\n")

    coefficients = lasso.coef_

    # Identify features with coefficients close to zero
    threshold = 0.01  # Set a threshold
    features_to_remove = np.where(np.abs(coefficients) < threshold)[0]
    print(f"Features to remove (indices): {features_to_remove}")

    # Optionally, remove those features from your dataset
    X_train_selected = np.delete(X_train_poly, features_to_remove, axis=1)
    X_test_selected = np.delete(X_test_poly, features_to_remove, axis=1)

    # predict quality based on X_train
    y_pred_train = lasso.predict(X_train_poly)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    print(f"Training RMSE Score (close to 0 = better):\t{train_rmse:.6f}")
    print(f"Training R-Squared Score (close to 1 = better):\t{train_r2:.6f}\n")
    
    # predict quality based on X_test
    y_pred_test = lasso.predict(X_test_poly)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    print(f"Testing RMSE Score (close to 0 = better):\t{test_rmse:.6f}")
    print(f"Testing R-Squared Score (close to 1 = better):\t{test_r2:.6f}\n")

    return lasso.intercept_, lasso.coef_, X_train_selected, X_test_selected
        
'''Call main'''
if __name__ == "__main__":
    main()
