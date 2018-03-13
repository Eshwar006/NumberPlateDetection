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
def find_if_close(cnt1, cnt2):
    row1, row2 = cnt1.shape[0], cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            dist = np.linalg.norm(cnt1[i] - cnt2[j])
            if abs(dist) < 10:
                return True
            else:
                return False
            

def contour_extraction(cropped, image, op,k):
    im2, contours, hierarchy = cv2.findContours(cropped, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(cropped, contours, -1, (0,255,0), 3)
    size = image.shape[:2]
    H = size[0]
    W = size[1]
    if not os.path.exists(op):
        os.makedirs(op)
        
    for (i,c) in enumerate(contours):
        (x,y,w,h) = cv2.boundingRect(c)
        #print(x,y,w,h)
        if w < W/4 and w*h < image.size/4 and w > W/20 and h > H/20:
            k += 1
            cv2.imwrite(op+"/"+str(k) + str(i) + ".png", image[y:y+h, x:x+w])
            cv2.rectangle(image, (x,y), (x+w,y+h), (0, 255, 0), 1)
            


#Read image
def function(inputpath, outputpath,k):
    print(inputpath, outputpath)
    originalimage = cv2.imread(inputpath)

    cv2.imshow('image', originalimage)
    thresholdedimage = preprocessing(originalimage)
    #cv2.imshow('thresholded image', thresholdedimage)
    angle, cropped = skew_angle(thresholdedimage)
    contour_extraction(cropped, originalimage, outputpath,k)
    print("Angle is ",angle)
    #cv2.imshow('cropped', cropped)
    cv2.imshow('Extracted image', originalimage)
    #cv2.imshow('thresh1', th1)

def run():
    path = sys.argv[1]
    outputpath = sys.argv[2]
    print(path, outputpath)
    #file_count = len([f for f in os.walk(path).next()[2] if f[-4:] == ".png"])
    #print(file_count)
    for i in range(20):
        ip = path + "/" + str(i+1) +".png"
        #op = outputpath + "/" + str(i+1)
        op = outputpath
        function(ip,op,i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

run()
