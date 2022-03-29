import cv2
import numpy as np

def neast_interpolation(img):
    h,w,c = img.shape
    empty_img = np.zeros((750,750,c),np.uint8)
    srch = 750 / h
    srcw = 750 / w

    for i in range(750):
        for j in range(750):
            x = int(i / srch)
            y = int(j / srcw)
            empty_img[i,j] = img[x,y]
    return empty_img
img = cv2.imread(r"lenna.png")
zoom = neast_interpolation(img)
cv2.imshow('img',img)
cv2.imshow("nerast_inter",zoom)
cv2.waitKey(0)
cv2.destroyAllWindows()