import numpy as np
import matplotlib.pyplot as plt 

def computeCost(X,y,theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/2*m)*np.sum(np.square(predictions - y))
    return cost

def computeGradientDescent(X,y,theta,learningRate,numberOfIterations):
    m = len(y)
    costHistory = np.zeros(numberOfIterations)

    for iteration in range(numberOfIterations):
        predictions = X.dot(theta)
        error = predictions - y
        theta = theta - (learningRate/m)*X.T.dot(error)
        costHistory[iteration] = computeCost(X,y,theta)
    return theta,costHistory


np.random.seed(0)
X_train = np.random.rand(100,3) * 10
Y_train = 2 + 3*X_train[:,0] + + 4*X_train[:,1] + 5*X_train[:,2] +  np.random.rand(100)

np.random.seed(0)
X_test = np.random.rand(50,3) * 10
Y_test = 2 + 3*X_test[:,0] + + 4*X_test[:,1] + 5*X_test[:,2] +  np.random.randn(50)

X_train = np.concatenate((np.ones((len(X_train), 1)), X_train), axis=1)
X_test = np.concatenate((np.ones((len(X_test), 1)), X_test), axis=1)

theta = np.zeros(X_train.shape[1])

numOfIterations = 1000
learningRate = 0.01

theta, costHistory = computeGradientDescent(X_train,Y_train,theta,learningRate,numOfIterations)

# Print the learned parameters
print("Learned parameters:")
print("Theta0 =", theta[0])
print("Theta1 =", theta[1])
print("Theta2 =", theta[2])
print("Theta3 =", theta[3])

plt.scatter(X_train[:, 1], Y_train, marker='o', c='b', label='Training Data')
plt.plot(X_train[:, 1], X_train.dot(theta), 'r-', label='Multiple Linear Regression (Training)')
plt.xlabel('X1')
plt.ylabel('y')
plt.legend()
plt.title('Multiple Linear Regression - Training Data')

# Plot the test data and the learned regression line
plt.figure()
plt.scatter(X_test[:, 1], Y_test, marker='o', c='g', label='Test Data')
plt.plot(X_test[:, 1], X_test.dot(theta), 'r-', label='Multiple Linear Regression (Test)')
plt.xlabel('X1')
plt.ylabel('y')
plt.legend()
plt.title('Multiple Linear Regression - Test Data')

plt.show()
