# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 14:50:16 2022

@author: HP
"""

import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import random
import pickle


datadir="D:/Anaconda/Python/ML/ML Odev1.1/Dataset2"
categories=['Car','Plane']
for category in categories:
    path=os.path.join(datadir,category)
    for img in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array,cmap="gray")
        plt.show()
        break
    break
print(img_array)
print(img_array.shape)

training_data=[]
class_nums=[]
    
def create_training_data():
    for category in categories:
        path=os.path.join(datadir,category)
        
        for img in os.listdir(path):
            img_array=cv2.imread(os.path.join(path,img))
            training_data.append(img_array)
            if category == "Plane":
                class_nums.append(0)
            else:
                class_nums.append(1)
            
    return img_array
            
image_array = create_training_data()
for i in training_data:
        print(i.shape)
from PIL import Image
new_process = list()
for i in training_data:
     image= Image.fromarray(i)
     image = image.resize((75,75))
     image = image.convert('L')
     new_process.append(np.array(image))
asd=np.array(new_process)
print(asd)
image = Image.fromarray(asd[-1])
print(class_nums[-1])
image.show()


X=asd.reshape(asd.shape[0],75*75)
y=np.array(class_nums)
y = y.reshape(y.shape[0],1)
data=np.hstack((X,y))
data.dump('data.pkl')
data=np.savetxt("data.csv", data, delimiter=",")






