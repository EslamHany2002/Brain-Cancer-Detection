import cv2
from filters import Process as pr

image = cv2.imread('Brain-Cancer-Detection/Datasets/yes/Y117.jpg')
im = pr.skull_del(image)
i = pr.thresh(im)
img = pr.skull_del(i)
ig = pr.segment1(img)


pr.img_diff(image,ig)
pr.img_plot(image,ig)