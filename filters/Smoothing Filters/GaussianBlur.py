import cv2
import numpy as np


img = cv2.imread('Tr-me_0080.jpg',1)
gaus = cv2.GaussianBlur(img,(7,7),2)


cv2.imshow('original',img)
cv2.imshow('GaussianBlur' , gaus)

cv2.waitKey(0)