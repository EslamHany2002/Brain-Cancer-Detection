import cv2
import numpy as np
img_org = cv2.imread('Tr-me_0080.jpg')

sharp_filter = np.array([[-1,-1,-1],
                         [-1,9,-1],
                         [-1,-1,-1]])
sharp_image = cv2.filter2D(img_org,-1,sharp_filter)
cv2.imshow('original',img_org)
cv2.imshow("sharp_image",sharp_image)

cv2.waitKey(0)
cv2.destroyAllWindows()