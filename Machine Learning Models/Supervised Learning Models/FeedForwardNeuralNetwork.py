import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers

#Load the Dataset 
digits = load_digits()
X,y = digits.data, digits.target

#Standardize the features - Standardizing means unit variance and zero mean
scalar = StandardScaler()
X = scalar.fit_transform(X)

#Split the Dataset into training and testing datasets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Create a Feed Forward Neural Network model with two Hidden Layers
model = keras.Sequential([
    layers.Dense(64,activation='relu',input_dim = X.shape[1]),
    layers.Dense(32,activation='relu'),
    layers.Dense(16,activation='relu'),
    layers.Dense(10,activation='softmax')
])

#Compile the Model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#Train the Model
history = model.fit(X_train,y_train,epochs=50,batch_size=16,validation_split=0.1)

#Evaluate the Model on Testing Data
test_loss, test_accuracy = model.evaluate(X_test,y_test)
print("Testing Accuracy:", test_accuracy)

# Plot training and validation accuracy over epochs
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Save the Neural Network graph to a file
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='model_graph.png')

