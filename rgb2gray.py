import numpy as np
import cv2

#读取图片
# img = cv2.imread(r"lenna.png")
# cv2.imshow('RGB_lean',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# #灰度化
#
# ###  方法一
# m,n = img.shape[:2]
#
# # 这里要注意一点，我所采用的图像的深度是uint8类型，
# # 由于uint8的范围太小，加和时一定会溢出，
# # 所以要提前一步进行数据转换，然后再变回uin8t类型。
#
# img_gray = np.zeros([m,n],dtype=img.dtype)  ##  数据类型需要与原图一致，这里为uint8
#
# for i in range(m):
#     for j in range(n):
#         p = img[i,j]
#         img_gray[i,j] = int(p[0] * 0.11 + p[1] * 0.59 + p[2] *0.3)
# print(img_gray)
# cv2.imshow('gray',img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

##方法二   直接利用cvtcolor函数进行操作，可以在不同颜色空间之间进行转换。

def easyWay(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow('img_gray',gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

easyWay(r"lenna.png")


# #方法三：分解原图像的三个通道，每个通道就会生成出一个灰度图像。
# # 这里也有两种方法，一个是利用opencv自带的函数split，
# # 第二种方法是单独遍历出每一个通道，建立新的图像。
#
# def BGR2Gray(filename):
#     img = cv2.imread(filename)
#     m = img.shape[0]
#     n = img.shape[1]
#
#     b = np.zeros([m,n],img.dtype)
#     g = np.zeros([m,n],img.dtype)
#     r = np.zeros([m,n],img.dtype)
#
#     b,g,r = cv2.split(img)
#
#     cv2.imshow('b',b)
#     cv2.imshow('g', g)
#     cv2.imshow('r', r)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# BGR2Gray(r"lenna.png")
#
# def bgr2gray(filename):
#     img = cv2.imread(filename)
#     m = img.shape[0]
#     n = img.shape[1]
#
#     # b = np.zeros([m,n],img.dtype)
#     # g = np.zeros([m,n],img.dtype)
#     # r = np.zeros([m,n],img.dtype)
#
#     b = img[:, :, 0]
#     g = img[:, :, 1]
#     r = img[:, :, 2]
#
#     cv2.imshow('b',b)
#     cv2.imshow('g', g)
#     cv2.imshow('r', r)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
#
# bgr2gray(r"lenna.png")