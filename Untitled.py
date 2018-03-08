#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 15:28:49 2018

@author: eshwarmannuru
"""

import cv2
#import numpy as np
#from PIL import Image

#Read image
image = cv2.imread('2.jpeg')
#cv2.imshow('Image', image)
#Convert to grayscale
grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(grayscaleImage,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#cv2.imshow('Thresh', thresh)

#Contours
contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
edged = cv2.Canny(thresh, 30, 200)

#cv2.imshow('Edged', edged)

(cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]


for cnt in contours:
    print(cv2.contourArea(cnt))
    if cv2.contourArea(cnt) > 50 and cv2.contourArea(cnt) < 1700:
        [x,y,w,h] = cv2.boundingRect(cnt)
        print('w=',w,'h=',h)
        if  h >= 10:
            cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
cv2.imshow('Final Image', image)

cv2.waitKey(0)
cv2.destroyAllWindows()



'''
#Apply Dilation and erosion to remove noise in the image
kernel = np.ones((1,1), np.uint8)
img = cv2.dilate(grayscaleImage, kernel, iterations=1)
#cv2.imshow('dilated', img)
img = cv2.erode(img, kernel, iterations=1)
#cv2.imshow('erode', img)
cv2.imshow('noiseless', img)


#Apply Thresholding to get a binary image
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


cv2.imshow('thresholded', img)

cv2.waitKey()
'''