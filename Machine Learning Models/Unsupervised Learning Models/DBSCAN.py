import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

# Generate Synthetic Data
n_samples = 1000
X, _ = make_moons(n_samples=n_samples, noise=0.5, random_state=42)

# Split the Dataset into training and testing Dataset
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Apply DBSCAN on Training Dataset
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_train = dbscan.fit_predict(X_train)

# Apply DBSCAN on Testing Dataset
labels_test = dbscan.fit_predict(X_test)

# Plot the Clusters for Training and Testing Dataset
plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
core_mask_train = dbscan.core_sample_indices_
plt.scatter(X_train[:, 0], X_train[:, 1], c=labels_train, cmap='viridis', s=50, alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data Clustering')
plt.colorbar()
plt.scatter(X_train[core_mask_train, 0], X_train[core_mask_train, 1], c='black', marker='x', s=100, label='Core Points')
plt.scatter(X_train[labels_train == -1, 0], X_train[labels_train == -1, 1], c='gray', s=50, label='Noise Points')
plt.legend()

plt.subplot(1, 2, 2)
core_mask_test = dbscan.core_sample_indices_
plt.scatter(X_test[:, 0], X_test[:, 1], c=labels_test, cmap='viridis', s=50, alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Testing Data Clustering')
plt.colorbar()
plt.scatter(X_test[core_mask_test, 0], X_test[core_mask_test, 1], c='black', marker='x', s=100, label='Core Points')
plt.scatter(X_test[labels_test == -1, 0], X_test[labels_test == -1, 1], c='gray', s=50, label='Noise Points')
plt.legend()

plt.tight_layout()
plt.show()
