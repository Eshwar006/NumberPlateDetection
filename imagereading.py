import numpy as np
import cv2
import sys
##import matplotlib.pyplot as plt


##---- Reading image ------- ##
img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
#IMREAD_COLOR = 1
#IMREAD_UNCHANGED = -1
#cv2.line(img, (0,0), (250,250), (255,255,255), 15)

laplacian = cv2.Laplacian(img, cv2.CV_64F)
edges = cv2.Canny(img, 10,10)

cv2.imshow('image', img)
cv2.imshow('laplacian', laplacian)
cv2.imshow('edges', edges)

cv2.waitKey(0)
cv2.destroyAllWindows()

