import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

# Load the Dataset
iris = load_iris()
X = iris.data

# Split the Dataset into training and testing Dataset
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Concatenate the Training and Testing Data
X_combined = np.vstack((X_train, X_test))

# Create t-SNE Object
tSNE = TSNE(n_components=2, perplexity=30, random_state=42)

# Fit and transform the Combined Data to 2D
X_combined_tSNE = tSNE.fit_transform(X_combined)

# Separate the transformed data back into Training and Testing sets
X_train_tSNE = X_combined_tSNE[:len(X_train)]
X_test_tSNE = X_combined_tSNE[len(X_train):]

# Plot the Training and Testing Data in 2D
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_train_tSNE[:, 0], X_train_tSNE[:, 1], cmap='viridis', alpha=0.6)
plt.title("t-SNE Visualization Training Data")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")

plt.subplot(1, 2, 2)
plt.scatter(X_test_tSNE[:, 0], X_test_tSNE[:, 1], cmap='viridis', alpha=0.6)
plt.title('t-SNE Visualization - Testing Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')

plt.tight_layout()
plt.show()
