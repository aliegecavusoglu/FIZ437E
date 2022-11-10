# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 22:49:22 2022

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings( "ignore" )

def train_test_split(X,y,size):

    num_data = X.shape[0]
    num_test = int(num_data*size)

    idxs = np.arange(num_data)
    np.random.shuffle(idxs)

    test_idxs = idxs[:num_test]

    train_idxs = idxs[num_test:]

    X_train = X[train_idxs]
    X_test = X[test_idxs]

    y_train = y[train_idxs]
    y_test = y[test_idxs]

    return X_train,X_test,y_train,y_test

class LogisticRegression():
    def __init__(self,learning_rate,iterations):
        self.learning_rate=learning_rate
        self.iterations=iterations
    def fit(self,X,Y):
        self.m,self.n=X.shape
        self.W=np.zeros(self.n)
        self.b=0
        self.X=X
        self.Y=Y
        
        for i in range(self.iterations):
            self.update_weights()
        return self
    
    def update_weights(self):
        A = 1 / ( 1 + np.exp( - ( self.X.dot( self.W ) + self.b ) ) )
          
        # calculate gradients        
        tmp = ( A - self.Y.T )        
        tmp = np.reshape( tmp, self.m )        
        dw = np.dot( self.X.T, tmp ) / self.m         
        db = np.sum( tmp ) / self.m 
        self.W=self.W-self.learning_rate*dw
        self.b=self.b-self.learning_rate*db
    
    def predict(self,X):
        Z=1/(1+np.exp(-(X.dot(self.W)+self.b)))
        Y=np.where(Z>0.5,1,0)
        return Y


data=np.load("data.pkl",allow_pickle=True)

X = data[:,:-1]
y = data[:,-1]       

X_train,X_test,y_train,y_test = train_test_split(X, y, 0.1)
    
def main():
    data=np.load("data.pkl",allow_pickle=True)

    X = data[:,:-1]
    y = data[:,-1]       

    X_train,X_test,y_train,y_test = train_test_split(X, y, 0.1)
    model=LogisticRegression(learning_rate=0.1, iterations=1000)
    model.fit(X_train,y_train)
    Y_pred_train=model.predict(X_train)
    Y_pred_test=model.predict(X_test)
    correctly_classified_train=0
    correctly_classified_test=0
    count_train=0
    count_test=0

    for count_train in range(np.size(Y_pred_train)):
        if y_train[count_train]==Y_pred_train[count_train]:
            correctly_classified_train=correctly_classified_train+1
        count_train=count_train+1
    
    for count_test in range(np.size(Y_pred_test)):
        if y_test[count_test]==Y_pred_test[count_test]:
            correctly_classified_test=correctly_classified_test+1
        count_test=count_test+1
    
    print( "Accuracy on training set by our model       :  ", (correctly_classified_train / count_train ) * 100 )
    print( "Accuracy on test set by our model       :  ", (correctly_classified_test / count_test ) * 100 ) 
    
main() 
    
    
    
    
    
    
    