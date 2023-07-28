import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def computeCost(X,y,theta):
    m = len(y) #This is the size of the training data set
    predictions = X.dot(theta) #This is matrix multiplication of Features with the Weight 'w' that is wX
    cost = (1/(2*m)) * np.sum(np.square(predictions-y)) #This is computing the Cost for each iteration that is 1 training example
    return cost

def computeGradientDescent(X,y,theta,learningRate,numOfIterations):
    m = len(y) #This is the size of the training data
    costHistory = np.zeros(numOfIterations) #This is an array which will store the cost for each of the training example

    for iteration in range(numOfIterations): #For the total number of training examples
        predictions = X.dot(theta) #Predicts the output for the particualar training example in the loop
        errors = predictions - y #Calculates the error that is predicted value - actual value in the training data set
        theta = theta - (learningRate/m)* X.T.dot(errors) #Calculates theta which is the weight 
        costHistory[iteration] = computeCost(X,y,theta) #This is an array which will store cost for all the training examples
    
    return theta,costHistory

#Generate Random Training Set 
#np.random.seed(0)
#X_train = np.random.rand(100,1) #100,1 is the dimensions of the array that is 100 examples of trainig set with 1 feature
#Y_train = 2 + 3*X_train + np.random.rand(100,1) #This is just arbitrary formuale for generating Y values of the training data 

#Generate Random Testing Set
#np.random.rand(50,1) 
#X_test = np.random.rand(50,1)
#Y_test = 2 + 3*X_test + np.random.rand(50,1)

df = pd.read_csv('/Users/jainamdoshi/Desktop/Datasets/SimpleLinearRegresisonDataset(Years of Exp vs Salary).csv')

# Extract the input feature and target variable
X = df['YearsExperience'].values.reshape(-1, 1)
y = df['Salary'].values.reshape(-1, 1)

# Divide the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Feature Scaling
mean = np.mean(X_train,axis = 0)
std = np.std(X_train,axis=0)
X_train_scaled = (X_train - mean)/std
X_test_scaled = (X_test - mean)/std


X_train_scaled = np.concatenate((np.ones((len(X_train_scaled), 1)), X_train_scaled), axis=1) #This will be adding the column of 1's as the first column of the matrix to match the matrix dimensions
X_test_scaled = np.concatenate((np.ones((len(X_test_scaled), 1)), X_test_scaled), axis=1)



#Initialize theta (Weights) with zeroes
theta = np.zeros((2,1)) #Initiating the theta values (w as theta1 and b as theta2)

#Set Hyperparameters 
numOfIterations = 1000
learningRate = 0.01

theta, costHistory = computeGradientDescent(X_train_scaled,y_train,theta,learningRate,numOfIterations)

# Print the learned parameters
print("Learned parameters:")
print("Theta0 =", theta[0][0])
print("Theta1 =", theta[1][0])

#Plot the training data 
plt.scatter(X_train_scaled[:,1],y_train,marker='o',c='b',label = 'Training Data') #This will be plotting the training points in the data set 
plt.plot(X_train_scaled[:,1],X_train_scaled.dot(theta),'r-',label = 'Linear Regression (Training)') #This will be plotting the prediction line of the training points in the data set 
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Simple Linear Regression - Training Data')


#PLot the Test data

plt.figure()
plt.scatter(X_test_scaled[:,1],y_test,marker = 'o',c = 'b', label = "Testing Data") #This will be plotting the testing points in the data set 
plt.plot(X_test_scaled[:,1],X_test_scaled.dot(theta),'r-',label = 'Linear Regression (Testing)') #This will be plotting the prediction line of the testing points in the data set
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Simple Linear Regression - Testing Data')

plt.show()

