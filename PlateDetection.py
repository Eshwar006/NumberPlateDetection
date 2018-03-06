import cv2
import sys
import numpy as np
import Preprocess

#Reading grayscale image and converting it into binary scale image
img = cv2.imread('1.png')
height, weight, numChannels = img.shape

imgGrayscalescene = np.zeros((height,weight,1), np.int8)
imgThreshscene = np.zeros((height,weight,1), np.int8)
imgContours = np.zeros((height,weight,3), np.int8)

#Preprocess
imgGrayscaleScene, imgThreshScene = Preprocess.preprocess(img) 

cv2.imshow('0', img)
cv2.imshow('1a', imgGrayscalescene)
cv2.imshow('1b', imgThreshscene)
cv2.imshow('1c', imgContours)


#ret, thresh_img = cv2.threshold(img, 100,255, cv2.THRESH_BINARY)

#cv2.imshow('thresh_img', thresh_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


#Get all the connected regions and group them together
