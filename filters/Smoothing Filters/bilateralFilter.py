import cv2
import numpy as np


img = cv2.imread('Tr-me_0080.jpg',1)
bilateralFilter = cv2.bilateralFilter(img,9,75,75)


cv2.imshow('original',img)
cv2.imshow('bilateralFilter', bilateralFilter)
cv2.waitKey(0)