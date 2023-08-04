import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

#Load the Dataset
iris = load_iris()
X = iris.data

#Split the Dataset into training and testing Dataset
X_train,X_test = train_test_split(X,test_size=0.2,random_state=42)

#Perform Hierarchial Clustering on Training Data
k = 3
agg_clustering = AgglomerativeClustering(n_clusters=k)
train_labels = agg_clustering.fit_predict(X_train)
train_centroids = np.array([X_train[train_labels == i].mean(axis=0) for i in range(k)])

#Predict Clustering for Testing Data
test_labels = agg_clustering.fit_predict(X_test)
test_centroids = np.array([X_test[test_labels == i].mean(axis=0) for i in range(k)])

# Plot the training data with cluster centroids
plt.scatter(X_train[:, 0], X_train[:, 1], c=train_labels, cmap='viridis', s=50, alpha=0.6)
plt.scatter(train_centroids[:, 0], train_centroids[:, 1], c='red', marker='x', s=100, label='Train Centroids')
plt.xlabel('Sepal length')
plt.ylabel('Sepal Width')
plt.title('Hierarchical Clustering on Training Data')
plt.legend()
plt.show()

# Plot the testing data with their own centroids
plt.scatter(X_test[:, 0], X_test[:, 1], c=test_labels, cmap='viridis', s=50, alpha=0.6)
plt.scatter(test_centroids[:, 0], test_centroids[:, 1], c='blue', marker='x', s=100, label='Test Centroids')
plt.xlabel('Sepal length')
plt.ylabel('Sepal Width')
plt.title('Hierarchical Clustering on Testing Data')
plt.legend()
plt.show()

