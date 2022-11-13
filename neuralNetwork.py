# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 23:59:37 2022

@author: HP
"""

import numpy as np



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

class neuralNetwork():
    def __init__(self):
        self.input=1
        self.output=1
        self.hiddenUnits=2
        np.random.seed(1)
        self.w1=np.random.randn(self.input,self.hiddenUnits)
        self.w2=np.random.randn(self.hiddenUnits,self.output)
    
    
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    
    
    
    def forwardPropogation(self,X):
        self.z2=np.dot(self.w1.T,X)
        self.a2=self.sigmoid(self.z2)
        self.z3=np.dot(self.w2.T,self.a2)
        self.a3=self.sigmoid(self.z3)
        return self.a3
    
    
    def Loss(self, predict, y):
        m = y.shape[0]
        logprobs = np.multiply(np.log(predict), y) + np.multiply((1 - y), np.log(1 - predict))
        loss = - np.sum(logprobs) / m
        return loss
    
    def sigmoidPrime(self, z):
        return self.sigmoid(z)*(1-self.sigmoid(z))
    
    def backwardPropagation(self, X, y):
        predict = self.forwardPropogation(X)
        m = X.shape[0]
        delta3 = predict - y
        dz3 = np.multiply(delta3, self.sigmoidPrime(self.z3))
        self.dw2 = (1/m)*np.sum(np.multiply(self.a2, dz3), axis=1).reshape(self.w2.shape)
        
        
        delta2 = delta3*self.w2*self.sigmoidPrime(self.z2)
        self.dw1 = (1/m)*np.dot(X.T, delta2.T)
        return self.dw1,self.dw2
        
        
    def update(self, learning_rate=1):
        self.w1 = self.w1 - learning_rate*self.dw1
        self.w2 = self.w2 - learning_rate*self.dw2
        
        
        
    def train(self, X, y, iteration=100):
        for i in range(iteration):
            y_hat = self.forwardPropogation(X)
            loss = self.Loss(y_hat, y)
            self.backwardPropagation(X,y)
            self._update()
            if i%10==0:
                print("loss: ", loss)
                
    def predict(self, X):
        y_hat = self.forwardPropogation(X)
        y_hat = [1 if i[0] >= 0.5 else 0 for i in y_hat.T]
        return np.array(y_hat)
    
    def score(self, predict, y):
        cnt = np.sum(predict==y)
        return (cnt/len(y))*100






def main():

    data=np.load("data.pkl",allow_pickle=True)

    X = data[:,:-1]
    y = data[:,-1]       

    X_train,X_test,y_train,y_test = train_test_split(X, y, 0.1)
    model=neuralNetwork()
    model.train(X_train,y_train)
    Y_pred_test=model.predict(X_test)
    accuracy_test=model.score(Y_pred_test,y_test)
    Y_pred_train=model.predict(X_train)
    accuracy_train=model.score(Y_pred_train,y_train)

    print(accuracy_train)
    print(accuracy_test)
    
    
main()    
        
    
    
    
        