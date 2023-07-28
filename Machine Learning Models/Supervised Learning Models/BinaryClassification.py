import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def computeCost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    cost = (-1 / m) * (y.T.dot(np.log(h)) + (1 - y).T.dot(np.log(1 - h)))
    return cost

def computeGradientDescent(X, y, theta, learningRate, numberOfIterations):
    m = len(y)
    costHistory = np.zeros(numberOfIterations)

    for iteration in range(numberOfIterations):
        h = sigmoid(X.dot(theta))
        gradient = (1/m) * X.T.dot(h - y)
        theta = theta - learningRate * gradient
        costHistory[iteration] = computeCost(X, y, theta)
    
    return theta, costHistory

def plotDecisionBoundary(X,y,theta):
    X1_min, X1_max = X[:,0].min() - 0.1, X[:,0].max() + 0.1
    X2_min, X2_max = X[:,1].min() - 0.1, X[:,1].max() + 0.1
    xx1,xx2 = np.meshgrid(np.linspace(X1_min,X1_max,100),np.linspace(X2_min,X2_max,100)) 
    xx = np.column_stack((np.ones(xx1.ravel().shape),xx1.ravel(),xx2.ravel()))
    Z = sigmoid(xx.dot(theta))
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, cmap=plt.cm.Paired)
    plt.scatter(X[:, 1], X[:, 2], c=y.ravel(), cmap=plt.cm.Paired)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()


# Generate random training set
np.random.seed(0)
X_train = np.random.rand(100, 2)
y_train = np.random.randint(0, 2, size=(100, 1))

# Initialize theta with zeros
theta = np.zeros((X_train.shape[1]+1, 1)) # +1 for bias term

# Set Hyperparameters
learningRate = 0.01
numberOfIterations = 1000

# Feature Scaling 
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train_scaled = (X_train - mean) / std

# Add bias column to X_train
X_train_scaled = np.concatenate((np.ones((len(X_train_scaled), 1)), X_train_scaled), axis=1)

theta, costHistory = computeGradientDescent(X_train_scaled, y_train, theta, learningRate, numberOfIterations)

# Plot cost history
plt.plot(costHistory)
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History')
plt.show()

#Plot Decision Boundary for Training Data
plotDecisionBoundary(X_train_scaled,y_train,theta)

# Generate random test set
X_test = np.random.rand(50, 2)
y_test = np.random.randint(0, 2, size=(50, 1))

# Perform feature scaling on test set
X_test_scaled = (X_test - mean) / std

# Add bias column to X_test
X_test_scaled = np.concatenate((np.ones((len(X_test_scaled), 1)), X_test_scaled), axis=1)

# Make predictions on test set
predictions = sigmoid(X_test_scaled.dot(theta))
predictions = (predictions >= 0.5).astype(int)

# Calculate accuracy
accuracy = np.mean(predictions == y_test) * 100
print(f"Accuracy: {accuracy}%")


#Plot Decision Boundary for Training Data
plotDecisionBoundary(X_test_scaled,y_test,theta)