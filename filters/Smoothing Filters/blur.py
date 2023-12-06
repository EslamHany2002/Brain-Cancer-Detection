import cv2



img = cv2.imread('Tr-me_0080.jpg',1)
averaging = cv2.blur(img,(11,11))
gaus = cv2.GaussianBlur(img,(7,7),2)
med = cv2.medianBlur(img,5)
bilateralFilter = cv2.bilateralFilter(img,9,75,75)
cv2.imshow('original',img)
cv2.imshow('averaging',averaging)
cv2.imshow('Median Blur', med)
cv2.imshow('GaussianBlur' , gaus)
cv2.imshow('bilateralFilter', bilateralFilter)
cv2.waitKey(0)
