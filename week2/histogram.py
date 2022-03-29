import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
calcHist—计算图像直方图
函数原型：calcHist(images, channels, mask, histSize, ranges, hist=None, accumulate=None)
images：图像矩阵，例如：[image]
channels：通道数，例如：0
mask：掩膜，一般为：None
histSize：直方图大小，一般等于灰度级数
ranges：横轴范围
'''
img =cv2.imread('lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)



# hist = cv2.calcHist([gray],[0],None,[256],[0,256])
# plt.figure()#新建一个图像
# plt.title("Grayscale Histogram")
# plt.xlabel("Bins")#X轴标签
# plt.ylabel("# of Pixels")#Y轴标签
# plt.plot(hist)
# plt.xlim([0,256])#设置x坐标轴范围
# plt.show()

##法2
plt.figure()
plt.hist(gray.ravel(), 256)
plt.show()

#彩色图像直方图
image = cv2.imread("lenna.png")
chans = cv2.split(image)
colors = ("b","g","r")
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan,color) in zip(chans,colors):
    hist = cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color = color)
    plt.xlim([0,256])
plt.show()
