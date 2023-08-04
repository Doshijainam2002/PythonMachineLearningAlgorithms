import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

#Load the Dataset
iris = load_iris()
X = iris.data

#Split the Dataset into Training and Testing Dataset
X_train,X_test = train_test_split(X,test_size=0.2,random_state=42)

#Peform Gaussian Mixture Model on the Training Data
k = 3
gmm = GaussianMixture(n_components=k)
train_labels = gmm.fit_predict(X_train)
train_probabilities = gmm.predict_proba(X_train)

#Predict Clusters and Probabilities for Testing Data
test_labels = gmm.fit_predict(X_test)
test_probabilities = gmm.predict_proba(X_test)

# Plot the training data with GMM clusters and centers
plt.scatter(X_train[:, 0], X_train[:, 1], c=train_labels, cmap='viridis', s=50, alpha=0.6)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=100, label='GMM Centers')
plt.xlabel('Sepal length')
plt.ylabel('Sepal Width')
plt.title('Gaussian Mixture Model Clustering on Training Data')
plt.legend()
plt.show()

# Plot the testing data with GMM clusters and centers
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_labels, cmap='viridis', s=50, alpha=0.6)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='red', marker='x', s=100, label='GMM Centers')
plt.xlabel('Sepal length')
plt.ylabel('Sepal Width')
plt.title('Gaussian Mixture Model Clustering on Testing Data')
plt.legend()
plt.show()
