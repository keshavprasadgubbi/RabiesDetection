#!/usr/bin/env python
# coding: utf-8

# In[39]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import random 

DATADIR = "Data/Train/"
CATEGORIES = ["Background","Image","Partial"]


def create_training_data():
    Training_Data = []
    img_size = 50
    X = []
    y = []
    for categories in CATEGORIES:
        Path = os.path.join(DATADIR,categories) # gives the path to the images; Data/Train/Image or Background
        print(Path)
        class_num = CATEGORIES.index(categories) # numbers the classes
        print(class_num)
        try:
            for img in os.listdir(Path): #gives the images from the path
                img_array = cv2.imread(os.path.join(Path,img),cv2.IMREAD_GRAYSCALE) 
                Training_Data.append([img_array,class_num])
                #plt.imshow(img_array,cmap='gray')
                #plt.show()
                #break
        except Exception as e:
            pass
    random.shuffle(Training_Data) # do it before conversion to array as list is mutable
    
    for features,labels in Training_Data:
        X.append(features)
        y.append(labels)
    
    X = np.array(X).reshape(-1,img_size,img_size,1) # converts it to arrays with shape (number of images,dim of image)
    y = np.array(y)
    print(X.shape)
    X = X/255.0 # this is normalization of only the images
    return X,y

X,y = create_training_data()


def Test_Data():
    img_size = 50 # fix the size of the image
    Path = "Data/Test"
    Test = []
    image_names = []
    try:
        for img in os.listdir(Path): #gives the images from the path
            img_array = cv2.imread(os.path.join(Path,img),cv2.IMREAD_GRAYSCALE) 
            Test.append(img_array)
            image_names.append(img)
            #plt.imshow(img_array,cmap='gray')
            #plt.show()
            #break
    except Exception as e:
        pass
    #print(Test.shape)
    Test = np.array(Test).reshape(-1,img_size,img_size,1)
    Test = Test/255.0
    #print(Test.shape)
    return Test, image_names 





