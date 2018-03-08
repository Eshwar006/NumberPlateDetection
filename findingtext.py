#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  8 17:42:45 2018
http://blog.ayoungprogrammer.com/2013/01/equation-ocr-part-1-using-contours-to.html/
@author: eshwarmannuru
"""


import cv2
import numpy as np

def preprocessing(image):
    blur = cv2.GaussianBlur(image, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 75, 10)
    return th

#Deskewing the text angle
def deSkew(thresh, image):
    coords = np.column_stack(np.where(thresh > 0))
    cv2.imshow('coords', coords)
    rect  = cv2.minAreaRect(coords)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    
    cv2.drawContours(image, [box], 0, (0, 255, 255))
    print(rect[2])
    angle = rect[2]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    print(w,h)
    center = (w // 2, h // 2)
    size = (int(rect[1][1]), int(rect[1][0]))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    cropped = cv2.getRectSubPix(image, size, center)
    cv2.imshow('Cropped', cropped)
    #cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return angle, rotated



#Read image
originalimage = cv2.imread('2.jpeg',0)
thresholdedimage = preprocessing(originalimage)
angle, rotated = deSkew(thresholdedimage, originalimage)

print(angle)
cv2.imshow('thresh', thresholdedimage)
cv2.imshow('Original Image',originalimage)
#cv2.imshow('rotated', rotated)



#th1 = cv2.bitwise_not(image, th)


#cv2.imshow('thresh1', th1)

cv2.waitKey(0)
cv2.destroyAllWindows()


