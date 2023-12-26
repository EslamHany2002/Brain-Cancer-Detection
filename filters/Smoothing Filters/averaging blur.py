import cv2
import numpy as np


img = cv2.imread('Y2.jpg',1)
averaging = cv2.blur(img,(11,11))


cv2.imshow('original',img)
cv2.imshow('averaging',averaging)
cv2.imwrite("blur3.jpg",averaging)

cv2.waitKey(0)