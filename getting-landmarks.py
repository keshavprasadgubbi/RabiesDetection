
# # Generating Landmarks
# 
# ## Local Landmark of Neuron
# * Step1: Obtain the tiles that are eitehr single neurons and/or stitched partial ones taht eligible to generate neurons
# * Step2: Obtain the central location of a given tile, based on thresholding and centre of mass of image which is the landmark needed : (x,y)   
# 
# ## Global Landmark of Neuron
# * Step3: In order to obtain global landmarks, get the respective tilenumber and also global coordinates of the tile. 
# * Step4: The local coordinates need to be subtracted from the (x2,y2) of the global coordinates: landmark_x = x2 - cx, landmark_y = y2 - cy
# * Step5: Add all the landmarks into a separate  text file

# ## Step2

import cv2 as cv
import numpy as np
source_path = '/Users/keshavaprasad/Desktop/image/Classification_model/singleneurontiles/3000.tif'
img = cv2.imread(source_path,0)
#print(img.shape)

import matplotlib.pyplot as plt
plt.imshow(img,cmap='gray')
plt.show()

# only thresholding + otsu + find contours
ret1,th1 = cv.threshold(img,127,255,cv.THRESH_BINARY+cv.THRESH_OTSU) #ret1 is the thresholding value and th1 is the thresholded image
#th1
contours,hierarchy = cv2.findContours(th1, 1, 2)
cnt = contours[0]
(cx,cy),radius = cv2.minEnclosingCircle(cnt)
center = (int(cx),int(cy))
radius = int(radius/10)
cv2.circle(img,center,radius,(0,255, 0), -1)
print(cx,cy)

import matplotlib.pyplot as plt
plt.imshow(th1,cmap='gray')
plt.show()

# calculating landmarks
#Need to get value of x2, y2 here which is the global coordinates of the tile! these are the global coordinates!
#landmark_x = x2 - cx # is the global landmark of the particular neuron
#landmark_y = y2 - cy

landmarks = [] 
landmarks.extend(list(zip(landmark_x,landmark_y))) #containing a tuple of landmarks

import csv

with open('landmarks.csv','wb') as out:
    csv_out=csv.writer(out)
    csv_out.writerow([landmark_x,landmark_y])
    for row in data:
        csv_out.writerow(row)

#ret,thresh = cv2.threshold(img,127,255,0)
#contours,hierarchy = cv2.findContours(thresh, 1, 2)
#cnt = contours[0]

#print(len(contours))
#print(ret)
#print(thresh)
#print(contours[0])
#print(hierarchy)
#M = cv2.moments(cnt)
#print( M)
#cx = int(M['m10']/M['m00'])
#cy = int(M['m01']/M['m00'])
#print("cx:",cx,",cy:",cy) # cx and cy is the local coordinate of the tile

#(cx,cy),radius = cv2.minEnclosingCircle(cnt)
#center = (int(cx),int(cy))
#radius = int(radius/10)
#cv2.circle(img,center,radius,(0,255, 0), -1)
#print(cx,cy)

#cv2.circle(img, (cx, cy), 1, (0, 0, 255), -1)

