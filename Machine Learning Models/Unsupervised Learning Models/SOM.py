import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.datasets import make_blobs

# Generate synthetic data
n_samples = 1000
n_features = 2
n_clusters = 4
X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# Normalize the data
X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# SOM hyperparameters
n_neurons = 10
learning_rate = 0.1
epochs = 100

# Initialize SOM weights randomly
som_weights = np.random.rand(n_neurons, n_features)

# Training process
for epoch in range(epochs):
    for sample in X_norm:
        distances = np.linalg.norm(sample - som_weights, axis=1)
        winner = np.argmin(distances)
        delta = learning_rate * (sample - som_weights[winner])
        som_weights[winner] += delta

# Testing process
X_test, _ = make_blobs(n_samples=200, n_features=2, centers=n_clusters + 1, random_state=42)
X_test_norm = (X_test - np.mean(X_test, axis=0)) / np.std(X_test, axis=0)

# Find the winning neuron for each test sample
winners = []
for sample in X_test_norm:
    distances = np.linalg.norm(sample - som_weights, axis=1)
    winner = np.argmin(distances)
    winners.append(winner)

winners = np.array(winners)

# Plotting
plt.scatter(X_test[:, 0], X_test[:, 1], c=winners, cmap='rainbow')
plt.scatter(som_weights[:, 0], som_weights[:, 1], marker='x', color='black')
plt.title('SOM Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.colorbar()
plt.show()
