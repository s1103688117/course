import cv2
import numpy as np

# img = cv2.imread(r"lenna.png")

# m,n = img.shape[:2]
#
# img_gray = np.zeros([m,n],img.dtype)
#
# for i in range(m):
#     for j in range(n):
#         p = img[i,j]
#         img_gray[i,j] = int(p[0] * 0.11 + p[1] * 0.59 + p[2] *0.3)
#
# cv2.imshow('gray',img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# 法1
# for i in range(m):
#     for j in range(n):
#         if img_gray[i,j] > 128:
#             img_gray[i,j] = 255
#         else:
#             img_gray[i,j] = 0
# cv2.imshow('binary',img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 方案二
# img_binary = np.where((img_gray) > 128, 255,0)  #int32会报错，应该为uint8
# print(img_binary.dtype)
# img_binary = np.uint8(img_binary)
# cv2.imshow('binary',img_binary)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#方案3
img = cv2.imread(r"lenna.png")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_binary = np.where((img_gray) > 128, 255,0)
img_binary = np.uint8(img_binary)
cv2.imshow('binary',img_binary)
cv2.waitKey(0)
cv2.destroyAllWindows()