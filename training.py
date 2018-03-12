#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 10:19:27 2018
@author: eshwarmannuru
OCR Training
"""
import sys
import cv2
import pytesseract
from PIL import Image

filename = '/home/eshwarmannuru/Desktop/NumberPlateDetection/results-images/1/2.png'
img = cv2.imread(filename)
text = pytesseract.image_to_string(Image.open(filename))
print(text)

