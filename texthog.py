#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 18:45:50 2018
@author: eshwarmannuru

Detecting given pattern or image is character or not
"""
import sys
import cv2
import os
import numpy as np

path = '/home/eshwarmannuru/Desktop/NumberPlateDetection/result/1/1.png'
path = sys.argv[1]
im = cv2.imread(path)

#print(im.shape)

im = np.float32(im) / 255.0

gx = cv2.Sobel(im, cv2.CV_32F, 1, 0, ksize = 1)
gy = cv2.Sobel(im, cv2.CV_32F, 0, 1, ksize = 1)

mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
cv2.imshow("Mag",mag)

hog = cv2.HOGDescriptor()

img = hog(im)


"""
cv2.imshow("Image",im)

from pytesseract import image_to_string
from PIL import Image

print(image_to_string(Image.open(path)))

cv2.waitKey(0)
cv2.destroyAllWindows()
"""

