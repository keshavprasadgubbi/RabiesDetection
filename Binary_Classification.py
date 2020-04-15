#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from Data_Preparation import *
from keras.utils import to_categorical


# In[2]:


X,y = create_training_data()


# In[3]:


#def binary_classification_model():
model = Sequential()
model.add(Conv2D(64,(3,3),padding = 'same',input_shape=(X.shape[1:]),kernel_initializer = 'he_normal'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(64,(3,3),padding = 'same',kernel_initializer = 'he_normal'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.7))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu")) # added a new activation layer

model.add(Dense(3))
model.add(Activation('softmax')) # chaging from sigmoid to softmax to include more classes

#filepath="weights.best.hdf5"
ada = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss ="categorical_crossentropy", optimizer = 'adam', metrics = ["accuracy"])
y_train=to_categorical(y, num_classes=3)

model.summary()
lr_model_history=model.fit(X,y_train,batch_size=10,epochs= 10,validation_split=0.1)


# In[4]:


fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(np.sqrt(lr_model_history.history['loss']), 'r', label='training')
ax.plot(np.sqrt(lr_model_history.history['val_loss']), 'b' ,label='validation')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Loss', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)

# Plot the accuracy
fig, ax = plt.subplots(1, 1, figsize=(10,6))
ax.plot(np.sqrt(lr_model_history.history['accuracy']), 'r', label='training')
ax.plot(np.sqrt(lr_model_history.history['val_accuracy']), 'b' ,label='validation')
ax.set_xlabel(r'Epoch', fontsize=20)
ax.set_ylabel(r'Accuracy', fontsize=20)
ax.legend()
ax.tick_params(labelsize=20)


# In[57]:


#import SimpleITK as sitk
#import os
#import glob
#reader = sitk.ImageFileReader()
#test_image_path = '/Users/keshavaprasad/Desktop/image/Classification_model/Data/Test/*'


# In[5]:


img_size = 50 # fix the size of the image
Path = "Data/Test"
Testing_image = []# list of testing images
image_name = []# list of image names
try:
    for img in os.listdir(Path): #gives the images from the path
        img_array = cv2.imread(os.path.join(Path,img),cv2.IMREAD_GRAYSCALE) # read every image in gray scale from the given path
        Testing_image.append(img_array)
        image_name.append(img)
        #plt.imshow(img_array,cmap='gray')
        #plt.show()
        #break
except Exception as e:
    pass


# In[8]:


#print(Testing_image[8])
#print(image_name[8])

if image_name[8]== '.DS_Store':
    print(Testing_image.pop(8))
    print(image_name.pop(8))


# In[61]:


#print(Testing_image.pop(8))
#print(image_name.pop(8))


# In[9]:


print(len(Testing_image))
print(Testing_image[8].shape)
print(len(image_name))
image_name


# In[10]:


Test_image = np.array(Testing_image).reshape(-1,img_size,img_size,1)#make an array of every element of list from Testing_image and then reshape them
#Test_image = Test_image/255.0


# In[11]:


#print(Test_image[8].shape)
#print(Test_image[8])


# In[12]:


scores = model.predict_classes(Test_image)
scores


# In[14]:


for i in range(len(Test_image)):
    print("Image:",image_name[i],"with score:",scores[i])    


# In[30]:


#need the images of class1 to be appended in a separate list
sn_list  = [] # contains list of filenames of single neurons
sn_images = [] # conatins tiles of single neurons #(actual images itself)
for i in range(len(Test_image)):
    if scores[i]==1:
        #print(Test_image[i])
        #print("Image ",image_name[i])
        sn_images.append(Test_image[i])
        sn_list.append(image_name[i])


# In[31]:


partial_list = []
partial_images = []
for i in range(len(Test_image)):
    if scores[i]==2:
        #print(Test_image[i])
        #print("Image ",image_name[i])
        partial_images.append(Test_image[i])
        partial_list.append(image_name[i])
    
#partial_list  


# In[32]:


print(len(partial_list))


# In[ ]:


'''# Display these single neurons
import matplotlib.pyplot as plt
plt.imshow(b[16],cmap='gray')
plt.show()
'''


# In[32]:


'''scores = []
for file in glob.glob(test_image_path):
    reader.SetFileName(file)
    a = reader.Execute()
    b = sitk.GetArrayFromImage(a)
    #b.shape
    scores.append(model.predict_classes(b[:,:,0].reshape(1,50,50,1)))
    #print(file,"scores:",scores)
'''


# In[52]:


#Test_D, img_names = Test_Data()
# Test_D is the image itself while img_names is a variable containing list of names of images
#print(img_names)
#print("*************")
#print(Test_D)
#scores = model.predict_classes(Test_D)
#print(Test_D.shape)
#for i in range(len(Test_D)):
#     print("Image:",img_names[i],"with score:",scores[i])


# In[55]:


#print(img_names)
#os.listdir()


# In[42]:


#print(type(scores)) # scores is numpy.ndarray. so need to convert it into a list to access list methods!
#scores_list = scores.tolist()


# In[49]:


'''single_indexes = []
partial_indexes = []

partial_neuron_list = []

for i in scores_list:
    if i ==1:
        single = scores_list.index(i)
        single_indexes.append(single)
for l in scores_list:
    if l ==2:
        partial = scores_list.index(l)
        partial_indexes.append(partial)
for j in indexes:
    partial_neuron_list.append(Test_D[j])
for k in indexes:
    single_neuron_list.append(Test_D[k])
  '''  


# In[27]:


#model.save('Model_That_Worked_for_three_classes.h5')


# In[28]:


#from keras.models import load_model
#worked_model = load_model('Model_That_Worked_for_three_classes.h5')


# In[81]:


#os.path.exists('/Users/keshavaprasad/Desktop/image/Classification_model/Data/Test/2076.tif')


# In[61]:


#b[:,:,0].shape


# In[56]:


#a.GetDimension()


# In[103]:


#img_size = 50
#Test = np.array(b).reshape(-1,img_size,img_size,1)
#C = Test/255.0
#print(b.shape)


# In[120]:


#b[:,:,0].reshape(1,50,50,1)

