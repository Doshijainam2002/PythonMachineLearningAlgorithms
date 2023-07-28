import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('/Users/jainamdoshi/Desktop/Datasets/MLRdata.csv')

X = df[['V1', 'V2', 'V3', 'V4', 'V5', 'V6']]
Y = df[['V7']]

# Convert DataFrame to numpy arrays
X = X.values
Y = Y.values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Polynomial regression
degree = 2  # Degree of the polynomial equation
X_train_poly = np.column_stack([np.power(X_train[:, i], j) for i in range(X_train.shape[1]) for j in range(1, degree + 1)])
X_test_poly = np.column_stack([np.power(X_test[:, i], j) for i in range(X_test.shape[1]) for j in range(1, degree + 1)])

# Initialize theta with zeros
theta = np.zeros((X_train_poly.shape[1], 1))

# Set hyperparameters
learning_rate = 0.01
num_iterations = 1000

# Perform gradient descent
m = len(y_train)  # Number of training examples

for iteration in range(num_iterations):
    # Calculate predictions
    predictions = X_train_poly.dot(theta)

    # Calculate errors
    errors = predictions - y_train

    # Update theta using vectorized gradient descent
    gradient = (1 / m) * X_train_poly.T.dot(errors)
    theta -= learning_rate * gradient

# Make predictions
y_train_pred = X_train_poly.dot(theta)
y_test_pred = X_test_poly.dot(theta)

# Evaluate the model
mse_train = np.mean((y_train_pred - y_train) ** 2)
mse_test = np.mean((y_test_pred - y_test) ** 2)
print("Mean Squared Error (Train):", mse_train)
print("Mean Squared Error (Test):", mse_test)

# Plot the results
plt.scatter(X_train[:, 0], y_train, color='blue', label='Training Data')
plt.scatter(X_test[:, 0], y_test, color='green', label='Testing Data')
plt.plot(X_train[:, 0], y_train_pred, color='red', label='Polynomial Regression')
plt.xlabel('V1')  # Choose appropriate feature to plot
plt.ylabel('V7')  # Choose appropriate target variable
plt.legend()
plt.title('Polynomial Regression - Training and Testing Data')
plt.show()
