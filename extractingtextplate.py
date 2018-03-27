#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 15:50:15 2018

@author: eshwarmannuru
"""

#from skimage.io import imread
#from skimage.filters import threshold_otsu
#import matplotlib.pyplot as plt
import sys
import cv2
import numpy as np
import os


def preprocessing(path, op, k):
    
    # Reading imput image
    image = cv2.imread(path + "/" + k)
    cv2.imshow('Original image1', image)
    
    # Converting to gray image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #cv2.imshow('gray image2', img_gray)
    
    # Noise removal using bilateral filter
    noise_removal = cv2.bilateralFilter(img_gray, 9, 75, 75)
    #cv2.imshow('noise removal3', noise_removal)
    
    #Histogram equalisation for better results
    equal_histogram = cv2.equalizeHist(noise_removal)    
    #cv2.imshow('Histogram4', equal_histogram)
    #equal_histogram = noise_removal
    
    #Morphological opening using rectangular structure element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    morph_image = cv2.morphologyEx(equal_histogram, cv2.MORPH_OPEN, kernel, iterations = 15)
    #cv2.imshow('morph_image5', morph_image)
    
    # Image subtraction 
    sub_morph = cv2.subtract(equal_histogram, morph_image)
    #cv2.imshow('Subtracted Image 6', sub_morph)
    
    #Thresholding image
    ret, thresh_image = cv2.threshold(sub_morph, 0, 255, cv2.THRESH_OTSU)
    #cv2.imshow('Thresh_image 7', thresh_image)
    
    #Applying canny edge detection
    canny_image = cv2.Canny(thresh_image, 250, 255)
    canny_image = cv2.convertScaleAbs(canny_image)
    #cv2.imshow('Canny Image 8', canny_image)
    
    #Dilation to strengthen edges
    kernel = np.ones((3,3), np.uint8)
    dilated_image = cv2.dilate(canny_image, kernel, iterations=1)
    #cv2.imshow('Dilated image 9', dilated_image)
    
    #Finding Contours in image based on edges
    new,contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours= sorted(contours, key = cv2.contourArea, reverse = True)[:10]
    
    #screenCnt = None
    # loop over our contours
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.06 * peri, True)  # Approximating with 6% error
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        #cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 1)
        if len(approx) == 4:  # Select the contour with 4 corners
            #cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 1)
            #screenCnt = approx
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.imshow('Extracted image ', image[y:y+h, x:x+w])
            cv2.imwrite(op+"/"+k , image[y:y+h, x:x+w])
            break
        #final = cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
        # Drawing the selected contour on the original image
        #cv2.imshow("Image with Selected Contour",final)
        
        

    # Masking the part other than the number plate
    #cv2.imshow('Rect Image 10', image)
    #mask = np.zeros(img_gray.shape,np.uint8)
    #new_image = cv2.drawContours(mask,[screenCnt],0,255,-1,)
    #new_image = cv2.bitwise_and(image,image,mask=mask)
    #cv2.imshow("Final_image 11",new_image)

    # Histogram equal for enhancing the number plate for further processing
    #y,cr,cb = cv2.split(cv2.cvtColor(new_image,cv2.COLOR_RGB2YCrCb))
    # Converting the image to YCrCb model and splitting the 3 channels
    #y = cv2.equalizeHist(y)
    # Applying histogram equalisation
    #final_image = cv2.cvtColor(cv2.merge([y,cr,cb]),cv2.COLOR_YCrCb2RGB)
    # Merging the 3 channels
    #cv2.imshow("Enhanced Number Plate 12",final_image)
    # Display image
    cv2.waitKey() # Wait for a keystroke from the user
    


def main(ip, op):
    for filename in os.listdir(path):
        print(str(filename))
        preprocessing(ip, op, filename)
        
        
path = sys.argv[1]
op = sys.argv[2]
main(path, op)



