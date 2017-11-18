# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 14:43:38 2017

@author: Michal
"""
from task import *
# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense



# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
print(X_train)

# Define the scaler 
scaler = StandardScaler().fit(X_train)

# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)

model = Sequential()

# Add an input layer 
model.add(Dense(4, activation='relu', input_shape=(3,)))

# Add one hidden layer 
model.add(Dense(5, activation='relu'))

# Add an output layer 
model.add(Dense(1, activation='sigmoid'))

# Model output shape
model.output_shape

# Model summary
model.summary()

# Model config
model.get_config()

# List all weight tensors 
model.get_weights()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
                   
model.fit(X_train, y_train,epochs=8, batch_size=1, verbose=1)

y_pred = model.predict(X_test)
score = model.evaluate(X_test, y_test,verbose=1)
print(score)




