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

#Deskew the text and crop 
#def deskew(image):



<<<<<<< HEAD
def contour_extraction(cropped, image, outputpath):
=======
def contour_extraction(cropped, image, op):
>>>>>>> 9d8588fc1e2657c20836f9d2e3983880c652bf84
    contours, hierarchy = cv2.findContours(cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(cropped, contours, -1, (0,255,0), 3)
    size = image.shape[:2]
    H = size[0]
    W = size[1]
<<<<<<< HEAD
    k  = 0
    if not os.path.exists(outputpath):
        os.makedirs(outputpath)
    for (i,c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        #print(x,y,w,h)
        if w < W/4 and w*h < image.size/4 and w > W/20:
            k += 1
            cv2.imwrite(outputpath + "/"+str(k)+".png", image[y:(y+h), x:(x+w)])
=======
    if not os.path.exists(op):
        os.makedirs(op)

    k = 0
    for (i,c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        #print(x,y,w,h)
        if w < W/4 and w*h < image.size/4 and w > W/20 and h > H/20:
            k += 1
            cv2.imwrite(op+"/"+str(k) + ".png", image[y:y+h, x:x+w])
>>>>>>> 9d8588fc1e2657c20836f9d2e3983880c652bf84
            cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 1)
            


#Read image
<<<<<<< HEAD
def function(path, outputpath):
    originalimage = cv2.imread(path)
=======
def function(inputpath, outputpath):
    print(inputpath, outputpath)
    originalimage = cv2.imread(inputpath)

    cv2.imshow('image', originalimage)
>>>>>>> 9d8588fc1e2657c20836f9d2e3983880c652bf84
    thresholdedimage = preprocessing(originalimage)

    #cv2.imshow('thresholded image', thresholdedimage)

    angle, cropped = skew_angle(thresholdedimage)
<<<<<<< HEAD
    contour_extraction(cropped, originalimage, outputpath)
    print("Angle is ",angle)
    cv2.imshow('cropped', cropped)
    cv2.imshow('Extracted image', originalimage)
    #cv2.imshow('thresh1', th1)

def run():
    path = sys.argv[1]
    outputpath = sys.argv[2]
    for i in range(20):
        print("Preprocessing Image: ", i+1)
        ip = path+"/" + str(i+1) + ".png"
        op = outputpath + "/" + str(i+1)
        print(ip)
        function(ip, op)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

    

=======
    cv2.imshow('cropped', cropped)
    contour_extraction(cropped, originalimage, outputpath)

    cv2.imshow('Extracted image', originalimage)
    #cv2.imshow('thresh1', th1)

path = sys.argv[1]
outputpath = sys.argv[2]

file_count = len([f for f in os.walk(path).next()[2] if f[-4:] == ".png"])

for i in range(file_count):
    ip = path + "/" + str(i+1) +".png"
    op = outputpath + "/" + str(i+1)
    function(ip,op)


cv2.waitKey(0)
cv2.destroyAllWindows()
>>>>>>> 9d8588fc1e2657c20836f9d2e3983880c652bf84


