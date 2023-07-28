import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Take the first two features for simplicity and better visualization
X = X[:, :2]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the Gradient Boosting model
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)

# Train the model
gb_model.fit(X_train, y_train)

# Make predictions on the training and testing data
y_train_pred = gb_model.predict(X_train)
y_test_pred = gb_model.predict(X_test)

# Calculate the accuracy of the model
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Plot training and testing datasets
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolors='k', label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', s=100, label='Testing Data')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.title('Training and Testing Datasets')
plt.show()

# Plot the boosting graph
train_accuracy_list = []
test_accuracy_list = []
for y_pred_train in gb_model.staged_predict(X_train):
    train_accuracy_list.append(accuracy_score(y_train, y_pred_train))

for y_pred_test in gb_model.staged_predict(X_test):
    test_accuracy_list.append(accuracy_score(y_test, y_pred_test))

plt.figure(figsize=(10, 6))
plt.plot(np.arange(1, gb_model.n_estimators + 1), train_accuracy_list, label='Training Set Score', marker='o')
plt.plot(np.arange(1, gb_model.n_estimators + 1), test_accuracy_list, label='Testing Set Score', marker='o')
plt.xlabel('Number of Boosting Iterations')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Gradient Boosting Algorithm')
plt.show()
