import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Embedding, SimpleRNN, Dense, LSTM
from keras.models import Sequential
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_default_https_context = ssl._create_unverified_context


#Set the Parameters
max_features = 10000
max_length = 500
batch_size = 32

#Load the IMDb Dataset 
(X_train,y_train),(X_test,y_test) = imdb.load_data(num_words=max_features)

#Pad the sequences to ensure all the inputs have same length 
X_train = sequence.pad_sequences(X_train,maxlen=max_length)
X_test = sequence.pad_sequences(X_test,maxlen=max_length)

#Create the RNN Model
model = Sequential()
model.add(Embedding(max_features,32))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))

#Compile the Model
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

#Train the Model
model.fit(X_train,y_train,epochs=5,batch_size=batch_size,validation_data=(X_test,y_test))

#Evaluate the Model on the Test Data
loss, accuracy = model.evaluate(X_test,y_test,batch_size=batch_size)
print(f'Test accuracy: {accuracy:.4f}')