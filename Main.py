import cv2
from filters import Process as pr

image = cv2.imread('Datasets\Dataset 1\yes\Y1.jpg')
im = pr.skull_del(image)
i = pr.thresh(im)
img = pr.skull_del(i)
ig = pr.segment1(img)

cv2.imwrite("image2",ig)
pr.img_diff(image,ig)
pr.img_diff(image,im)
pr.img_plot(image,ig)