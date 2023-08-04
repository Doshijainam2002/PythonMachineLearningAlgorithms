import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Dataset
iris = load_iris()
X, y = iris.data, iris.target

# Take the first two features of the Dataset for simplicity and better visualization
X = X[:, :2]

# Divide the Dataset into Training and Testing Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN model with K=5
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the Model
knn_model.fit(X_train, y_train)

# Make Predictions
y_train_pred = knn_model.predict(X_train)
y_test_pred = knn_model.predict(X_test)

# Measure Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the model Accuracy
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Create a Meshgrid of points to plot decision boundaries
h = 0.2
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Plot the KNN decision boundary
plt.figure(figsize=(10, 8))
Z = knn_model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.3)

# Plot training and testing datasets
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolors='k', label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', s=100, label='Testing Data')

plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.title('K-Nearest Neighbors Decision Boundary with Training and Testing Datasets')

# Plot the KNN graph
k_values = np.arange(1, 31)  # Vary K from 1 to 30 (you can change the range as desired)
train_accuracy_list = []
test_accuracy_list = []

for k in k_values:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_train_pred = knn_model.predict(X_train)
    y_test_pred = knn_model.predict(X_test)
    train_accuracy_list.append(accuracy_score(y_train, y_train_pred))
    test_accuracy_list.append(accuracy_score(y_test, y_test_pred))

plt.figure(figsize=(10, 6))
plt.plot(k_values, train_accuracy_list, label='Training Set Score', marker='o')
plt.plot(k_values, test_accuracy_list, label='Testing Set Score', marker='o')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Accuracy')
plt.legend()
plt.title('K-Nearest Neighbors Algorithm')
plt.show()
