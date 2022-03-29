#双线性插值法
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import math
def bilinear(img,dstH,dstW):
    src_H,src_W,_=img.shape
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for dstX in range(dstH-1):
        for dstY in range(dstW-1):
            #源图像和目标图像几何中心的对齐
            srcX=(dstX+0.5)*(src_W/dstW)-0.5
            srcY=(dstY+0.5)*(src_H/dstH)-0.5
            #四个点坐标
            x1=round(srcX - 0.5)
            x2=round(srcX+0.5)
            y1=round(srcY-0.5)
            y2=round(srcY+0.5)
            #X轴方向线插
            fy1=(x2-srcX)*img[int(x1),int(y1)]+(srcX-x1)*img[int(x2),int(y1)]
            fy2=(x2-srcX)*img[int(x1),int(y2)]+(srcX-x1)*img[int(x2),int(y2)]
            #Y轴方向线插
            retimg[dstX,dstY]=(y2-srcY)*fy1+(srcY-y1)*fy2
    return retimg
im = cv2.imread('lenna.png')
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# im=Image.open('lenna.png')
# im_array = np.array(im)
image1 = bilinear(im, im.shape[0] * 2, im.shape[1] * 2)
# image1 = Image.fromarray(image1.astype('uint8')).convert('RGB')
# image1.show()
image2 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
cv2.imshow("tes",image2)
cv2.waitKey(0)
cv2.destroyAllWindows()



###双三次插值
from PIL import Image
import numpy as np
import math
# 产生16个像素点不同的权重
def BiBubic(x):
    x=abs(x)
    if x<=1:
        return 1-2*(x**2)+(x**3)
    elif x<2:
        return 4-8*x+5*(x**2)-(x**3)
    else:
        return 0
# 双三次插值算法
def BiCubic_interpolation(img,dstH,dstW):
    scrH,scrW,_=img.shape
    #img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=i*(scrH/dstH)
            scry=j*(scrW/dstW)
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            tmp=0
            for ii in range(-1,2):
                for jj in range(-1,2):
                    if x+ii<0 or y+jj<0 or x+ii>=scrH or y+jj>=scrW:
                        continue
                    tmp+=img[x+ii,y+jj]*BiBubic(ii-u)*BiBubic(jj-v)
            retimg[i,j]=np.clip(tmp,0,255)
    return retimg

im = cv2.imread('lenna.png')
image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

image2=BiCubic_interpolation(image,image.shape[0]*2,image.shape[1]*2)
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
cv2.imshow("res",image2)
cv2.waitKey(0)
cv2.destroyAllWindows()



###老师代码
'''
python implementation of bilinear interpolation
'''
def bilinear_interpolation(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)
    if src_h == dst_h and src_w == dst_w:
        return img.copy()
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
    for i in range(3):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # find the origin x and y coordinates of dst image x and y
                # use geometric center symmetry
                # if use direct way, src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # find the coordinates of the points which will be used to compute the interpolation
                #找到四个点坐标
                src_x0 = int(np.floor(src_x))
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)


                # calculate the interpolation  由于图像双线性插值只会用相邻的4个点，公式的分母都是1
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    return dst_img


if __name__ == '__main__':
    img = cv2.imread('lenna.png')
    dst = bilinear_interpolation(img, (700, 700))
    cv2.imshow('bilinear interp', dst)
    cv2.waitKey()