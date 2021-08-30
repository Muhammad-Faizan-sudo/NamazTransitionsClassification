import os
import pandas as pd
import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras import layers, models
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D

categories = ['1 Qayam to ruko','2 ruko to qouma','3 qouma to sajda','4 sajda to jalsa','5 jalsa to sajda','6 sajda to qayam','7 jalsa to RS','8 RS to LS']
dataDir = "/home/pi/program_transition/New folder"
training_data = []
angles = []

for category in categories:
    path = os.path.join(dataDir,category)
    class_num = categories.index(category)
    for transition in os.listdir(path):
        try:
            read_transition_data = pd.read_csv(os.path.join(path,transition),index_col=None, header=None, engine='python')
            training_data.append([read_transition_data, class_num])
        except Exception as e:
            pass

import random
random.shuffle(training_data)
'''
for sample in training_data: 
    print(sample)
'''
X = []
y = []
for features, label in training_data:
    X.append(features)
    y.append(label)
print("The total transition files = ",len(X))
'''
print(np.array(X,dtype = object).shape)
print(np.array(y).shape)
'''
from sklearn import preprocessing
trainingData=[]
for i in range(len(X)):
    x_new = np.round(preprocessing.normalize(X[i]),4)
    trainingData.append(x_new)
print(len(trainingData))
trainingData = np.array(trainingData,dtype = object)
y= np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(trainingData, y,train_size=0.8, test_size = 0.2, random_state = 0)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

model = Sequential()
model.add(Conv1D(30, kernel_size=(3),activation='relu',input_shape=(600,3),padding='same'))
model.add(MaxPooling1D((3),padding='same'))
model.add(Dropout(0.4))

model.add(Conv1D(15, kernel_size=(3), activation='relu',padding='same'))
model.add(MaxPooling1D((3),padding='same'))
'''
model.add(Conv1D(15, kernel_size=(3), activation='relu',padding='same'))
model.add(MaxPooling1D((3),padding='same'))
'''

model.add(Flatten())
model.add(Dense(50, activation='relu'))    
model.add(Dense(25, activation='relu'))    
model.add(Dense(15, activation='relu'))
model.add(Dense(8, activation='softmax'))

model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics = ['accuracy'])

X_train = np.asarray(X_train).astype(np.float64)

model.fit(X_train, y_train, epochs = 50)
X_test = np.asarray(X_test).astype(np.float64)

model.evaluate(X_test, y_test)
model.summary()
model.save('T1')
new_model = tf.keras.models.load_model('T1')
# Check its architecture
new_model.summary()