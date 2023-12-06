import cv2

image = cv2.imread('Tr-me_0080.jpg')
edge = cv2.Canny(image , 100 , 200)
cv2.imshow('original',image)
cv2.imshow("Image Edges ", edge)
cv2.waitKey(0)
cv2.destroyAllWindows