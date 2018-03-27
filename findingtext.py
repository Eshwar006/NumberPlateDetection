#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:42:45 2018
http://blog.ayoungprogrammer.com/2013/01/equation-ocr-part-1-using-contours-to.html/
@author: eshwarmannuru
"""

import cv2
import numpy as np
import sys
import os


def preprocessing(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10)
    th = cv2.bitwise_not(th)
    return th

#Finding skew angle
def skew_angle(thresh):
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = thresh.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    #cropped = cv2.getRectSubPix(rotated, size, center)
    #cv2.imshow('Cropped', cropped)
    #cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return angle, rotated            

def contour_extraction(cropped, image, op,k):
    im2, contours, hierarchy = cv2.findContours(cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(cropped, contours, -1, (0,255,0), 3)
    size = image.shape[:2]
    H = size[0]
    W = size[1]
    if not os.path.exists(op):
        os.makedirs(op)
    p = 0;
    for (i,c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        #print(x,y,w,h)
        #if w < W/4 and w*h < image.size/4 and w > W/20 and h > H/20:
            #cv2.imshow(str(k)+str(i), image[y:y+h, x:x+w])
        p += 1
        cv2.imwrite(op+"/"+str(p) + k , image[y:y+h, x:x+w])
        cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 1)
            
    cv2.imshow('Extracted image', image)
    cv2.waitKey(0)

#Read image
def function(inputpath, outputpath,k):
    print(inputpath, outputpath)
    originalimage = cv2.imread(inputpath + "/" + k)

    cv2.imshow('image', originalimage)
    thresholdedimage = preprocessing(originalimage)
    #cv2.imshow('thresholded image', thresholdedimage)
    angle, cropped = skew_angle(thresholdedimage)
    contour_extraction(cropped, originalimage, outputpath,k)


def function1(inputpath):
      
    image = cv2.imread(inputpath)
    # Converting to gray image
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Noise removal using bilateral filter
    noise_removal = cv2.bilateralFilter(img_gray, 9, 75, 75)
    #Morphological opening using rectangular structure element
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    morph_image = cv2.morphologyEx(noise_removal, cv2.MORPH_OPEN, kernel, iterations = 15)
    
    # Image subtraction 
    sub_morph = cv2.subtract(noise_removal, morph_image)
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
    contours= sorted(contours, key = cv2.contourArea, reverse = True)[:25]

    size = image.shape[:2]
    H = size[0]
    W = size[1]
    
    # loop over our contours
    for i,c in enumerate(contours):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.01* peri, True)
        (x,y,w,h) = cv2.boundingRect(c)
        if len(approx) == 4:
            (x,y,w,h) = cv2.boundingRect(c)
            #if w < W/2 and h < H:
                #cv2.imshow(str(i), image[y:y+h, x:x+w])
        if w < W and w*h < image.size/4:
            cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 1)
        #final = cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
        # Drawing the selected contour on the original image
        #cv2.imshow("Image with Selected Contour",final)
    
    # Merging the 3 channels
    cv2.imshow("Enhanced Number Plate",image)
    # Display image
    cv2.waitKey(0) # Wait for a keystroke from the user
    cv2.destroyAllWindows()
    
def run(path, op):
    for filename in os.listdir(path):
        print(str(filename))
        function(ip,op,filename)
    cv2.destroyAllWindows()

ip = sys.argv[1]
op = sys.argv[2]
run(ip, op)
#function1(path)



