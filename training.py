#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 11:57:59 2018

@author: eshwarmannuru
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense

def read_image(path):
    image = cv2.imread(path)
    print(cv2.image.size)
    return image

def resize(image):
    resized_image = cv2.resize(image, (100, 50))
    return resized_image

def scaling(X_train, X_test):
    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

def label_encoding(Y):
    labelencoder_X_1 = LabelEncoder()
    Y = labelencoder_X_1.fit_transform(Y)
    return Y

def split(X,Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    return X_train, X_test, Y_train, Y_test

def neural_network(ip_dim, op_dim, hid_dim):
    #Initializing Neural Network
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = hid_dim[0], init = 'uniform', activation = 'relu', input_dim = ip_dim))

    # Adding the hidden layers
    for i in range(1,len(hid_dim)):
        classifier.add(Dense(output_dim = hid_dim[i], init = 'uniform', activation = 'relu'))
    
    # Adding the output layer
    classifier.add(Dense(output_dim = op_dim, init = 'uniform', activation = 'sigmoid'))
    
    # Compiling Neural Network
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
    
def train(classifier, X_train, Y_train):    
    # Fitting our model 
    classifier.fit(X_train, Y_train, batch_size = 10, nb_epoch = 100)
    return classifier

def predict(classifier, X_test):
    # Predicting the Test set results
    y_pred = classifier.predict(X_test)
    return y_pred

def accuracy_calc(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(accuracy*100)








