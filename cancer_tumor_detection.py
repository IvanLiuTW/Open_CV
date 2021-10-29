#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  5 12:05:37 2021

@author: ivanliu
"""
# 利用正子斷層造影，檢測原生性的食道癌
# 正子藥劑經靜脈注射入人體後，經衰變的過程產生的正電子在體組織中運行不到1毫米的距離，
# 即與帶負電荷的電子撞擊相互抵消毀滅(物理學上稱互毀作用，annihilation)，互抵消毀滅後質量不見了，
# 於是以能量的方式放出來，而放出來的能量是以兩道(方向相反，呈180度)加馬射線呈現，
# 而每一道加馬射線的能量為511 仟電子伏特(keV)。正子斷層造影儀(即執行PET的儀器; 
# 又稱PET掃描儀)可同時偵測這些成對的加馬射線，利用電腦重組正子同位素在組織或器官內分佈的圖像。

import numpy as np
import matplotlib.pyplot as plt
import cv2
from os import listdir # to read image files in the folder

# To transform the OpenCV picture to matplotlib plot, and show the picture
def aidemy_imshow(name, img):
    b, g, r = cv2.split(img)   #將各種通道(B, G, R)的資料拆解(split)出來
    img = cv2.merge([r, g, b]) #依 R, G, B 的順序再次合併
    plt.title(name)            # 以 matplotlib 繪圖
    plt.imshow(img)            # 以 matplotlib 讀取圖片
    plt.show()                 # 以 matplotlib 顯示圖片

cv2.imshow2 = aidemy_imshow 
   
# In[Prerequiste]

### to read all of the original photos into a list

path = "/Users/ivanliu/Desktop/Projects/Project_6_cancer_tumor_detection/ESO-051原始"
# to read all the names of files in the path
file_names = [f for f in listdir(path) if not f.startswith('.')] # not f.startswith('.') -> to avoid reading hidden files

images = [cv2.imread(path+"/"+i) for i in file_names]

### get the positions of ROI(Region of Interest) on the picture

img_open = cv2.imread(path + "/I10.jpg")
# select ROI
r = cv2.selectROI(img_open)
cv2.destroyAllWindows()
# crop the images and write them to the target folder
target_path = "/Users/ivanliu/Desktop/Projects/Project_6_cancer_tumor_detection/ESO_cropped/"
for num, item in enumerate(images):
    imCrop = item[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    cv2.imwrite(target_path + str(num)+'.jpg', imCrop)


### the concept of cropping an image:
r    
# (84, 469, 837, 452)
#  x    y    w    h   

# 裁切區域的 x 與 y 座標（左上角）
# x = 84
# y = 469
# 裁切區域的長度與寬度
# w = 837
# h = 452
# 裁切圖片
# crop_img = img[y:y+h, x:x+w]


# In[To find the tumor in the cropped images]

# load the image and convert it to grayscale
img_open_49 = cv2.imread("/Users/ivanliu/Desktop/Projects/Project_6_cancer_tumor_detection/ESO_cropped/49.jpg") # the picture contains tumor
orig = img_open_49.copy() # make a copy because we will mark the circle on the original picture at last
gray_49 = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY) 

### the way to find the brightest "spot" in the picture, use cv2.minMaxLoc() method

# perform a naive attempt to find the (x, y) coordinates of
# the area of the image with the largest intensity value
(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(gray_49)
img_open_49 = orig.copy() 
cv2.circle(img_open_49, maxLoc, 10, (255, 0, 0), 2)
# circle(img, center, radius, color[, thickness[, lineType[, shift]]])

### However, we can apply cv2.GaussianBlur() with cv2.minMaxLoc()
# to find the brightest "area" in the picture, rather than the spot

# apply a Gaussian blur to the image
after_blur = cv2.GaussianBlur(gray_49, (7, 7), 0)

(minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(after_blur)
img_open_49 = orig.copy() 
cv2.circle(img_open_49, maxLoc, 12, (255, 0, 0), 2)
# display the results of our newly improved method
cv2.imshow("Test", img_open_49);cv2.waitKey(0);cv2.destroyAllWindows();cv2.waitKey(1)
# the image first shown as not responding, and it's closed after hitting a key

# In[]

# use argparse if intending to run the script through terminal
# import argparse
#  construct the argument parse and parse the arguments, use this if we want to run code through terminal
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", help = "path to the image file")
# ap.add_argument("-r", "--radius", type = int, help = "radius of Gaussian blur; must be odd")
# args = vars(ap.parse_args()) #ap.parse_args() 回傳變數的剖析

# In[ introduction of cv2.GaussianBlur() ]
# dst = cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType=BORDER_DEFAULT]]] )
# Parameter	Description
# src	    input image
# dst	    output image
# ksize	    Gaussian Kernel Size. [height width]. height and width should be odd and can have different values. 
            # If ksize is set to [0 0], then ksize is computed from sigma values.
# sigmaX	Kernel standard deviation along X-axis (horizontal direction).
# sigmaY	Kernel standard deviation along Y-axis (vertical direction). If sigmaY=0, then sigmaX value is taken for sigmaY
# borderType	Specifies image boundaries while kernel is applied on image borders. 
# Possible values are : cv.BORDER_CONSTANT cv.BORDER_REPLICATE cv.BORDER_REFLECT cv.BORDER_WRAP cv.BORDER_REFLECT_101 
# cv.BORDER_TRANSPARENT cv.BORDER_REFLECT101 cv.BORDER_DEFAULT cv.BORDER_ISOLATED

# In[in-class practice]

# In{}
imgopen_1.shape #(442, 840, 3)
# from b, g, r changed to r, g, b
def change(img):
    b, g, r = cv2.split(img)
    mat_img = cv2.merge([r,g,b])
    return mat_img

img01 = change(imgopen_1)
img02 = change(imgopen_2)
img03 = change(imgopen_3)
img04 = change(imgopen_4)
img05 = change(imgopen_5)
img06 = change(imgopen_6)
img07 = change(imgopen_7)
img08 = change(imgopen_8)
img09 = change(imgopen_9)
img01.shape #(442, 840, 3)

aff_matrix = cv2.getRotationMatrix2D((img01.shape[1]/2, img01.shape[0]/2), 180, 0.8)
#cv2.getRotationMatrix2D((旋轉中心點的x座標, 旋轉中心點的y座標), 選轉角度, 放大倍數)

img01_rotate = cv2.warpAffine(img01, aff_matrix, (img01.shape[1], img01.shape[0]))
img02_blur = cv2.GaussianBlur(img02, (49, 49), 0) #y軸未指定，因此沿用x軸標準差

mask = cv2.imread('mask.jpg',cv2.IMREAD_GRAYSCALE)
mask = cv2.resize(mask, (img03.shape[1], img03.shape[0]))
img03_masked = cv2.bitwise_and(img03, img03, mask=mask)
mask2 = cv2.bitwise_not(mask)	
img04_masked = cv2.bitwise_and(img04, img04, mask=mask2)

thr, img05_binary = cv2.threshold(img05, 192, 255, cv2.THRESH_TOZERO)

img06_convert = cv2.cvtColor(img06, cv2.COLOR_BGR2GRAY)	


plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=2.5, hspace=0.5)
# 調整小圖的位置，避免圖片、標籤重疊
# 下面畫出3張圖，一列3個共兩列
plt.subplot(331);plt.imshow(img09);plt.title("Original")
plt.subplot(332);plt.imshow(img01_rotate);plt.title("Rotate")
plt.subplot(333);plt.imshow(img02_blur);plt.title("Blur")
plt.subplot(334);plt.imshow(img03_masked);plt.title("bitwise_and")
plt.subplot(335);plt.imshow(img04_masked);plt.title("bitwise_not")
plt.subplot(336);plt.imshow(img05_binary);plt.title("THRESH_TOZERO")
plt.subplot(337);plt.imshow(img06_convert);plt.title("COLOR_BGR2GRAY")
plt.subplot(338);plt.imshow(img07);plt.title("Test")
plt.subplot(339);plt.imshow(img08);plt.title("Test")
plt.show()