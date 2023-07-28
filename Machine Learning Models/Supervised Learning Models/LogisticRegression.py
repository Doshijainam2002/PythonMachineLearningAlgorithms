import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split


def sigmoid(z):
    clipped_z = np.clip(z, -500, 500)  # Clip the values to prevent overflow/underflow (Handles the runtime error which is thrown for the exponential function)
    return 1 / (1 + np.exp(-clipped_z))


# Load the Framingham dataset
df = pd.read_csv('/Users/jainamdoshi/Desktop/Datasets/framingham.csv')

# Extract the features (X) and target (y)
X = df[['male', 'age', 'education', 'currentSmoker', 'cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']]
Y = df['TenYearCHD']

# Convert DataFrame to numpy arrays
X = X.values
Y = Y.values

# Add Bias term to the features (X)
X = np.c_[np.ones((X.shape[0], 1)), X]

# Divide the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Initialize the parameters theta with zeroes
theta = np.zeros(X_train.shape[1])

# Set hyperparameters
learningRate = 0.01
numOfIterations = 1000

# Perform Gradient Descent
for iteration in range(numOfIterations):
    predictions = sigmoid(X_train.dot(theta))

    gradient = np.dot(X_train.T, predictions - y_train) / len(y_train)

    # Update the parameters
    theta = theta - learningRate * gradient

# Compute the predictions for the training and testing data
X_train_pred = sigmoid(np.dot(X_train, theta))
X_test_pred = sigmoid(np.dot(X_test, theta))

plt.scatter(X_train[y_train == 0, 4], X_train[y_train == 0, 14], label='Class 0')
plt.scatter(X_train[y_train == 1, 4], X_train[y_train == 1, 14], label='Class 1')
plt.xlabel('Feature 13')
plt.ylabel('Feature 15')
plt.title('Training Data')
plt.legend()
plt.show()


plt.scatter(X_test[y_test == 0, 4], X_test[y_test == 0, 14], label='Class 0')
plt.scatter(X_test[y_test == 1, 4], X_test[y_test == 1, 14], label='Class 1')
plt.xlabel('Feature 13')
plt.ylabel('Feature 15')
plt.title('Testing Data')
plt.legend()
plt.show()

