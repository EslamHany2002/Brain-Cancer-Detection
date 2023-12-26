import cv2
import numpy as np


img = cv2.imread('Tr-me_0080.jpg',1)
med = cv2.medianBlur(img,5)


cv2.imshow('original',img)
cv2.imshow('Median Blur', med)

cv2.waitKey(0)