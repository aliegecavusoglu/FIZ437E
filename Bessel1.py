# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 22:32:32 2022

@author: HP
"""

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
import seaborn as sns
from scipy.optimize import curve_fit
from random import sample



v=0
x=8*np.random.rand(10,1)
y=sp.jv(v,x)+random.gauss(0,0.1)
poly_features=PolynomialFeatures(degree=8,include_bias=False)
# x_poly=poly_features.fit_transform(x)

Xtr,Xte,Ytr,Yte=train_test_split(x,y,test_size=0.2,train_size=0.8)
# reg=LinearRegression()
# reg.fit(x_poly,y)
Xtr=Xtr.reshape(-1,1)
Xte=np.linspace(0,9,100).reshape(-1,1)
polyreg=make_pipeline(PolynomialFeatures(8),Ridge(alpha=0.001))
polyreg.fit(Xtr,Ytr)
plot=polyreg.predict(Xte) 
# polyreg.fit(Ytr,Yte)
  
sns.set_theme()
plt.plot(x, y)
plt.scatter(Xtr,Ytr)   
plt.plot(Xte, plot, label="fit", color = "red")

plt.show()



# x_vals=np.linspace(0,8,10).reshape(-1,1)
# x_vals_poly=poly_features.transform(x_vals)
# y_vals=reg.predict(x_vals_poly)
# plt.scatter(x,y)
# plt.plot(x_vals,y_vals,color='red')
# plt.show
n=len(Xtr)
errorTrain = np.zeros(n)
errorTest = np.zeros(n)
errorTrain=np.asmatrix(errorTrain)
errorTest=np.asmatrix(errorTest)

for i in range(1, n):
        polyreg.fit(Xtr, Ytr)
        Ypredict_train = polyreg.predict(Xtr)
        Ypredict_test = polyreg.predict(Xte)
        errorTrain = ((Ypredict_train - Ytr).dot(Ypredict_train - Ytr)) / len(Ypredict_train)
        errorTest= ((Ypredict_test - Yte).dot(Ypredict_test -Yte)) / len(Ypredict_test)
#validation_error = np.mean(np.square(np.array(Ytr)-np.array(Yte)))
#print(validation_error)