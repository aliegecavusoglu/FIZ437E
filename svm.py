# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 19:04:39 2022

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
warnings.filterwarnings( "ignore" )

class SVM():
    
    def __init__(self,learningRate,iterations,lambda_):
        self.learningRate=learningRate
        self.iterations=iterations
        self.lambda_=lambda_
        
        
    def fit(self,X,Y):
        self.m,self.n=X.shape
        self.w=np.zeros(self.n)
        self.b=0
        self.X=X
        self.Y=Y
        
        for i in range(self.iterations):
            self.updateWeights()
            
    def updateWeights(self):
        y_label=np.where(self.Y<=0,-1,1)
        
        for index,x_i in enumerate(self.X):
            condition=y_label[index]*(np.dot(x_i,self.w)-self.b)>=1
            if(condition==True):
                dw=2*self.lambda_*self.w
                db=0
            else:
                dw=2*self.lambda_*self.w-np.dot(x_i,y_label[index])
                db=y_label[index]
                
            self.w=self.w-self.learningRate*dw
            self.b=self.b-self.learningRate*db
            
            
            
    def predict(self,X):
        o1=np.dot(X,self.w)-self.b
        predictedLabels=np.sign(o1)
        y_hat=np.where(predictedLabels<=-1,0,1)
        return y_hat



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


def main():
    data=np.load("data.pkl",allow_pickle=True)

    X = data[:,:-1]
    y = data[:,-1]       

    X_train,X_test,y_train,y_test = train_test_split(X, y, 0.1)
    # print(data.shape)
    classifier=SVM(learningRate=0.001,iterations=1000,lambda_=0.01)
    classifier.fit(X_train, y_train)
    Y_pred_train=classifier.predict(X_train)
    Y_pred_test=classifier.predict(X_test)
    correctly_classified_train=0
    count_train=0
    correctly_classified_test=0
    count_test=0

    for count_train in range(np.size(Y_pred_train)):
        if y_train[count_train]==Y_pred_train[count_train]:
            correctly_classified_train=correctly_classified_train+1
        count_train=count_train+1

    for count_test in range(np.size(Y_pred_test)):
        if y_test[count_test]==Y_pred_test[count_test]:
            correctly_classified_test=correctly_classified_test+1
        count_test=count_test+1
    
    print("Accuracy on training set by our model   : ",(correctly_classified_train/count_train)*100)
    print("Accuracy on test set by our model   : ",(correctly_classified_test/count_test)*100)
main()


            
            
            
            
            
            
            
            
            
            
            
                    