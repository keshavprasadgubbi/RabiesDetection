#!/usr/bin/env python
# coding: utf-8

# # Implementing the cropping function
# 
# ## The various sub-tasks involved are :
# 
# * Get shape of the big image : (p,q)
# 
# * Define the size of the tile : (a,b) = (50,50)
# 
# * Determine the number of tiles : n = (p,q)/(a,b); ensure all tiles are of same dimensions and discard the ones that arent - Might lose a bit of the tiles from the edges but thats okay!
# 
# * Crop the tiles sequentially in tiles of size (a,b)
# 
# * Append all these tiles into a separate list so that its easier to call them up by their index in order to stitch them back again later!
# 
# * Discard any tile that does not conform to the size (a,b)

#Import Stuff 
import os
import cv2
import math
import numpy as np

#Read the image
source_path = '/Users/keshavaprasad/Desktop/image/MG45_bs_S010_00/MG45_bs_S10_00.tif'
#source_path = '/Users/keshavaprasad/Desktop/image/S25/S25_00.tif'
image=cv2.imread(source_path)

# Image Basics
print(image.shape)
print(image.size)
# Get Shape
p = image.shape[0]# p
q = image.shape[1]#q
print("Height:",p,"Width:",q)

# Defining the dimensions of a tile 
a = b = 50
#************************
cropped_tiles = [] #list for storing all cropped tiles
vertices = [] #list for storing lists of vertex values
landmarks = []

# Cropping image Sequentially
i = j = cropped_tile_counter = correct_tile_counter = wrong_tile_counter = 0

# Defining vertices of tiles for cropping
for i in range(0,math.floor(p),a):#range(start, stop, step) # to get the resolution right!
    for j in range(0,math.floor(q),b):
        
        #Defining the cropping box
        #tile = image[j:j+b,i:i+a]#y1: y2, x1: x2
        y1 = j
        x1 = i
        y2 = j+b
        x2 = i+a
        tile = image[x1:x2,y1:y2] # making crops of each subimage and then storing in tile variable and then appended to cropped_tiles list
        vertex = [x1,x2,y1,y2] # list of vertex values of each tiles
        #print(vertex)
        cropped_tile_counter +=1
        
        # checking for tiles with wrong dimensions
        if tile.shape[0]!=50 or tile.shape[1]!= 50:
            wrong_tile_counter += 1
            #print("p:",tile.shape[0],"q:",tile.shape[1])
            print("x1:",i,"y1:",j,"x2:",x2,"y2:",y2)
        
        # Keeping only tiles with correct dimensions and append the vertex for each of these tiles
        if tile.shape[0] == 50 and tile.shape[1] == 50:
            cropped_tiles.append(tile)
            vertices.append(vertex)
            correct_tile_counter +=1

#len(cropped_tiles)
#len(vertices)
#for item in vertices:
#    print(item)

print("cropped_tile_counter:",cropped_tile_counter,"correct_tile_counter:",correct_tile_counter,"wrong_tile_counter:",wrong_tile_counter)
print("useful tiles:",(correct_tile_counter/cropped_tile_counter)*100,"%","wasted tiles:",(wrong_tile_counter/cropped_tile_counter)*100,"%")
#print(cropped_tiles[670].shape)


# The tile number and respective vertices values of the tile is stored in a dictionary!
# This will be useful later in order to obtain the landmarks of respeective tiles!
#cropping_dict = {"cropped_tile_number":[x1,x2,y1,y2]} #values of the key is stored in a list in order to store multiple values for each key!
#cropping_dict = {"cropped_tile_number":[x1,y1,x2,y2]}
#print(cropping_dict["cropped_tile_number"][0])
#print("cropped_tile_number:",cropped_tile_number,"with vertices:",cropping_dict["cropped_tile_number"])
#for x, y in cropping_dict.items():
        #    print(x, y)     

# Saving the tiles in a respective folder

Dest_Path = '/Users/keshavaprasad/Desktop/image/testingcrop_MG45_bs_S010_00/'
#Dest_Path = '/Users/keshavaprasad/Desktop/image/S25/'
for k in range(len(cropped_tiles)):
    pass
    #print("K:",k)
    #cv2.imwrite(str(k)+ ".tif", tiles[k])
    #cv2.imwrite(os.path.join(Dest_Path,str(k)+".tif"), tiles[k])

#import matplotlib.pyplot as plt
#plt.imshow(tile[13180],cmap='gray')
#plt.show()
#print(tiles[7100])


# Cropping image Sequentially
'''
i=j=x1=x2=y1=y2 = 0
tile_number = 0
#y1 = j*50
#x1 = i*50
#y2 = (j+1)*50
#x2 = (i+1)*50
#range(0, p, a)
for i in range(0,math.floor(p/.868),a):# to get the resolution right!
    #print( "i:",i)
    for j in range(0,math.floor(q/.868),b):
        #print ("j:",j)
        #Defining the cropping box
        #y1: y2, x1: x2
        y1 = j*50
        x1 = i*50
        y2 = (j+1)*50
        x2 = (i+1)*50
        
        #tile = image[j*50:50*(j+1),i*50:50*(i+1)]
        tile = image[y1:y2,x1:x2]
        #tile_number +=1
        #print(tile_number)
        print(tile.shape)
        
        
        #print("tile_number:",tile_number,"X1:",x1,"Y1:",y1,"X2:",x2,"Y2:",y2)
        if tile.shape[0] == 50 and tile.shape[1] == 50:
            tiles.append(tile)
            #tile_number +=1
            #print(tile_number)
            
'''

