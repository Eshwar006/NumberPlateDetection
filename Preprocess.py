# Preprocess.py
import cv2
import numpy as np
import math
import pytesseract
from PIL import Image

#Read image
image = cv2.imread('1.jpg')

#Convert to grayscale
grayscaleImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Apply Dilation and erosion to remove noise in the image
kernel = np.ones((1,1), np.uint8)
img = cv2.dilate(grayscaleImage, kernel, iterations=1)
#cv2.imshow('dilated', img)
img = cv2.erode(img, kernel, iterations=1)
#cv2.imshow('erode', img)
cv2.imshow('noiseless', img)


#Apply Thresholding to get a binary image
img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

result = pytesseract.image_to_string(img)
print(result)

cv2.imshow('thresholded', img)

cv2.waitKey()


