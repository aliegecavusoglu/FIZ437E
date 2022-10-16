# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 15:03:06 2022

@author: HP
"""

import numpy as np
import matplotlib.pyplot as plt

def train_test_divider(X,y,size):

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

def acurracy_score(y_true,y_pred):
    acc = sum(y_true==y_pred)/y_true.shape[0]
    return acc
def ecludian_distance(X,x_pr):
    distances = np.sum((X-x_pr)**2,axis=1)
    return distances

class KNN():
    def __init__(self,K):
        self.K = K
    
    def fit(self,X,y):
        self.X = X
        self.y = y
    
    def predict(self,X_pr):
        pred_labels = np.array([])
        for i in range(X_pr.shape[0]):


            distances =ecludian_distance(self.X,X_pr[i])
            
                
            idx = np.argsort(distances)[:self.K]
            lowest_distances = self.y[idx]
            label = np.bincount(lowest_distances).argmax()
            pred_labels = np.append(pred_labels,label)

        return pred_labels

data=np.load("data.pkl",allow_pickle=True)

X = data[:,:-1]
y = data[:,-1]       

X_train,X_test,y_train,y_test = train_test_divider(X, y, 0.1)


knn = KNN(K=3)
knn.fit(X_train,y_train)

train_preds = knn.predict(X_train)
test_preds = knn.predict(X_test)

print("Accuracy score for test for K = 3: {}".format(acurracy_score(y_test,test_preds)))
print("Accuracy score for train for K = 3: {}".format(acurracy_score(y_train,train_preds)))


knn = KNN(K=1)
knn.fit(X_train,y_train)

train_preds = knn.predict(X_train)
test_preds = knn.predict(X_test)

print("Accuracy score for test for K = 1: {}".format(acurracy_score(y_test,test_preds)))
print("Accuracy score for train for K = 1: {}".format(acurracy_score(y_train,train_preds)))

K_values = np.arange(1,4)

test_accs = list()
train_accs = list()
for i in K_values:
    knn = KNN(K=i)
    knn.fit(X_train,y_train)

    train_preds = knn.predict(X_train)
    test_preds = knn.predict(X_test)
    acc_test = acurracy_score(y_test,test_preds)
    acc_train = acurracy_score(y_train,train_preds)

    train_accs.append(acc_train)
    test_accs.append(acc_test)
    print("Accuracy score for test for K = {}: {}".format(i,acc_test))
    print("Accuracy score for train for K = {}: {}".format(i,acc_train))
    

plt.plot(K_values,train_accs,label="Train Data")
plt.plot(K_values,test_accs,label="Test Data")
plt.grid()
plt.legend()
plt.show()
