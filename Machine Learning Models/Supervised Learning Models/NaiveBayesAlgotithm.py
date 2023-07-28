import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load the Datasets
iris = load_iris()
X, y = iris.data, iris.target

# Take the first two features for simplicity and Better visualization
X = X[:, :2]

# Split the Dataset into training and testing Datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Naive Bayes Model
nb_model = GaussianNB()

# Train the Model
nb_model.fit(X_train, y_train)

# Make predictions
y_train_pred = nb_model.predict(X_train)
y_test_pred = nb_model.predict(X_test)

# Measure the Accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print the Accuracy scores
print("Training Accuracy:", train_accuracy)
print("Testing Accuracy:", test_accuracy)

# Create a meshgrid of points to plot a decision boundary
h = 0.2
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))  # Corrected the parenthesis
Z = nb_model.predict(np.c_[xx.ravel(), yy.ravel()])

# Plot the Naive Bayes Decision Boundary
plt.figure(figsize=(10, 8))
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap='viridis', alpha=0.3)

# Plot the Training and Testing Datasets
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', edgecolors='k', label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', marker='x', s=100, edgecolors='k', label='Testing Data')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend()
plt.title('Naive Bayes Decision Boundary with Training and Testing Datasets')
plt.show()

# Plot the Naive Bayes Graph
plt.figure(figsize=(10, 6))
class_labels = list(iris.target_names)
class_probabilities = nb_model.predict_proba(X_test)

for class_index, class_label in enumerate(class_labels):
    plt.plot(np.arange(1, len(X_test) + 1), class_probabilities[:, class_index], label=class_label)

plt.xlabel('Test Data Instances')
plt.ylabel('Probability')
plt.legend()
plt.title('Naive Bayes Probabilities for Each Class on Testing Data')
plt.show()
