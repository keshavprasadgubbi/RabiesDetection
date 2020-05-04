#!/usr/bin/env python
# coding: utf-8

# In[1]:


import SimpleITK as sitk
import glob
import os
from scipy.ndimage import distance_transform_cdt
import numpy as np


# In[15]:


path = '/Users/keshavaprasad/Desktop/image/Classification_model/singleneurontiles/*'
output_path = '/Users/keshavaprasad/Desktop/image/Classification_model/landmarks/'
#for file in glob.glob(path):
#    print(file)


# In[29]:


x_coordinates = [] #x-coordinate
y_coordinates = [] #y-coordinate   
z_coordinates = [] #y-coordinate 
coordinates = [] #x,y-coordinate put together
for file in glob.glob(path):
    #print(file)
    im = sitk.ReadImage(file)
    fil = sitk.OtsuThresholdImageFilter()
    fil.SetInsideValue(0)
    fil.SetOutsideValue(255)
    binim = fil.Execute(im)
    sitk.WriteImage(binim,output_path+os.path.basename(file))
    npimage = sitk.GetArrayFromImage(binim)
    #dists = distance_transform_cdt(npimage,return_distances=True)
    indsx,indsy = np.where(npimage>0)
    z=0
    #centroid = [indsy.mean(),indsx.mean()]
    x,y = [indsy.mean(),indsx.mean()]
    coordinates.append([x,y,z])
    #print(centroid)

    
#print(coordinates)


# In[26]:


def writeLandmarkCoords(filename, landmarkList):
  with open(filename, 'w') as file:
     file.write("# Avizo 3D ASCII 2.0 \n")
     file.write("define Markers ")
     file.write("{}".format(len(landmarkList)))
     file.write("\n")
     file.write("Parameters {\n")
     file.write("    NumSets 1,\n")
     file.write("    ContentType \"LandmarkSet\"\n")
     file.write("}\n")

     file.write("Markers { float[3] Coordinates } @1\n")

     file.write("# Data section follows\n")
     file.write("@1\n")
     for landmark in landmarkList:
         file.write("{}".format(landmark[0]))
         file.write(" ")
         file.write("{}".format(landmark[1]))
         file.write(" ")
         file.write("{}".format(landmark[2]))
         file.write("\n")


# In[27]:


writeLandmarkCoords(output_path+"first.landmarkAscii", coordinates)


# In[ ]:




