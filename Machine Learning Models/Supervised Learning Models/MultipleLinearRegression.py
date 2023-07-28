import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
from sklearn.model_selection import train_test_split

def computeCost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))
    return cost

def computeGradientDescent(X, y, theta, learningRate, numberOfIterations):
    m = len(y)
    costHistory = np.zeros(numberOfIterations)

    for iteration in range(numberOfIterations):
        predictions = X.dot(theta)
        error = predictions - y
        theta = theta - (learningRate / m) * X.T.dot(error)
        costHistory[iteration] = computeCost(X, y, theta)
    return theta, costHistory


df = pd.read_csv('/Users/jainamdoshi/Desktop/Datasets/MLRdata.csv')

# Extract the features and the target variable from the dataset
X = df[['V1','V2','V3','V4','V5','V6']]
Y = df[['V7']]

# Convert DataFrame to numpy arrays
X = X.values
Y = Y.values

# Divide the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Prepend a column of ones to the input feature matrices to account for the bias term
X_train = np.concatenate((np.ones((len(X_train), 1)), X_train), axis=1)
X_test = np.concatenate((np.ones((len(X_test), 1)), X_test), axis=1)

theta = np.zeros((X_train.shape[1], 1))  # Reshape theta to (7, 1)

numOfIterations = 1000
learningRate = 0.01

theta, costHistory = computeGradientDescent(X_train, y_train, theta, learningRate, numOfIterations)

# Print the learned parameters
print("Learned parameters:")
for i, theta_i in enumerate(theta):
    print(f"Theta{i} =", theta_i)

# Plot the line of regression for training data
plt.figure()
plt.scatter(X_train[:, 1], y_train, marker='o', c='b', label='Training Data')
plt.plot(X_train[:, 1], X_train.dot(theta), 'r-', label='Multiple Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Multiple Linear Regression - Training Data')

# Plot the line of regression for testing data
plt.figure()
plt.scatter(X_test[:, 1], y_test, marker='o', c='g', label='Test Data')
plt.plot(X_test[:, 1], X_test.dot(theta), 'r-', label='Multiple Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Multiple Linear Regression - Test Data')

plt.show()
