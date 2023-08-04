import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np  

# Load the Minst Dataset
(x_train,y_train), (x_test,y_test) = mnist.load_data()

# Pre-process the data
x_train = x_train.reshape(-1,28,28,1).astype('float32')/255.0
x_test = x_test.reshape(-1,28,28,1).astype('float32')/255.0

# Create a CNN Model
learning_rate = 0.001
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
model = keras.Sequential([
    keras.layers.Conv2D(64,(3,3),activation='relu',input_shape=(28,28,1)),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Conv2D(64,(3,3,), activation='relu'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPooling2D((2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128,activation='relu'),
    keras.layers.Dense(10,activation='softmax')
])

# Compile the Model
model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

# Data Augmentation 
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Fit the Data Augmentation to the Training Data
datagen.fit(x_train)

# Train the model with Data Augmentation and different batch size (e.g., 64)
batch_size = 64
steps_per_epoch = len(x_train) // batch_size
epochs = 5

train_generator = datagen.flow(x_train, y_train, batch_size=batch_size)

model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=(x_test, y_test))

# Load the provided input image for testing
input_image = cv2.imread('/Users/jainamdoshi/Desktop/Datasets/SampleImageCNN.png', cv2.IMREAD_GRAYSCALE)
input_image = cv2.resize(input_image, (28, 28)) / 255.0
input_image = np.expand_dims(input_image, axis=-1)
input_image = np.expand_dims(input_image, axis=0)

# Predict the number
prediction = model.predict(input_image)
predicted_number = np.argmax(prediction)

print("Predicted Number:", predicted_number)
