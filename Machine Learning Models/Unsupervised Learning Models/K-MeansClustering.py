import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

#Load the Dataset
iris = load_iris()
X = iris.data

#Split the Dataset into training and testing Dataset
X_train,X_test= train_test_split(X,test_size=0.2,random_state=42)

#Perform K-means Clustering on Training Data
k = 3
kMeans = KMeans(n_clusters=k)
kMeans.fit(X_train) 
train_labels = kMeans.labels_
train_centroids = kMeans.cluster_centers_

#Predict Clusters for Testing Data
test_labels = kMeans.predict(X_test)
test_centroids = kMeans.cluster_centers_

#Plot the Datapoints for Training Dataset
plt.scatter(X_train[:,0],X_train[:,1],cmap='viridis',c=train_labels,s=50,alpha=0.6)
plt.scatter(train_centroids[:,0],train_centroids[:,1],c='red',marker='x',s=100,label='Train Centroids')
plt.xlabel('Sepal length')
plt.ylabel('Sepal Width')
plt.title('K Means Clustering on Training Dataset')
plt.legend()
plt.show()

#Plot the Datapoints for Testing Dataset
plt.scatter(X_test[:,0],X_test[:,1],cmap='viridis',c=test_labels,s=50,alpha=0.6)
plt.scatter(test_centroids[:,0],test_centroids[:,1],c='blue',marker='x',s=100,label='Test Centroids')
plt.xlabel('Sepal length')
plt.ylabel('Sepal Width')
plt.title('K Means Clustering on Testing Dataset')
plt.legend()
plt.show()



