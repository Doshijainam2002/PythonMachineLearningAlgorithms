import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

# Load the Dataset
iris = load_iris()
X = iris.data

# Split the Dataset into training and testing Dataset
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Perform PCA on the Training Data
k = 2
pca = PCA(n_components=k)
X_train_pca = pca.fit_transform(X_train)

# Calculate Explained Variance for Training Data
train_explained_variance = pca.explained_variance_ratio_

# Transform the Testing Data using the trained PCA model
X_test_pca = pca.transform(X_test)

# Calculate Explained Variance for Testing Data
test_explained_variance = pca.explained_variance_ratio_

# Principal component indices (1, 2, ..., k)
principal_components = range(1, k+1)

# Plot the graph for Training and Testing Data
plt.plot(principal_components, train_explained_variance, label='Training Data', marker='o', color='blue', alpha=0.6)
plt.plot(principal_components, test_explained_variance, label='Testing Data', marker='o', color='green', alpha=0.6)
plt.xlabel("Principal Component")
plt.ylabel("Explained Variance Ratio")
plt.title("PCA: Explained Variance for Training and Testing Data")
plt.legend()
plt.grid(True)
plt.show()
