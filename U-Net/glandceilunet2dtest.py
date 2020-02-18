from __future__ import division
from unet2d.model_GlandCeil import unet2dModule
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf


def train():
    '''
    Preprocessing for dataset
    '''
    # Read  data set (Train data from CSV file)
    csvmaskdata = pd.read_csv("GlandsMask.csv")
    #print(csvmaskdata)
    csvimagedata = pd.read_csv("GlandsImage.csv")
    #print(csvimagedata)
    maskdata = csvmaskdata.iloc[:, :].values
    #print(maskdata)
    imagedata = csvimagedata.iloc[:, :].values
    #print(imagedata)
    #img = cv2.imread("Segment/Image/0.bmp")
    #cv2.imshow('result.jpg',img)
    #cv2.waitKey(1000)
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]
    #print(imagedata)
    #print(maskdata)
    unet2d = unet2dModule(512, 512, channels=1, costname="dice coefficient")
    #unet2d.train(imagedata, maskdata, "model/unet2dglandceil.pd",
    #             "GlandCeil_Unet/log", 0.0005, 0.8, 100000, 2)
    unet2d.train(imagedata, maskdata, "model/unet2dglandceil.pd",
                 "model/log", 0.001, 0.5, 10, 4)

def predict(): 
    tf.reset_default_graph()#清除当前默认图中堆栈，重置默认图，实现模型参数的多次读取
    root_path1="IMG-0001-00"
    root_path2="12-"
    num=191
    end_path=".jpg"
    sigle_path="_Single.jpg"
    total_path="_Total.jpg"
    
    for i in range(0,1):
        readRoot=root_path1+str(num)+end_path
        writeRoot_Single=root_path2+str(num)+sigle_path
#        writeRoot_Toatl=root_path2+str(num)+total_path    
        true_img = cv2.imread(readRoot, cv2.IMREAD_GRAYSCALE)
#        cv2.imshow("111",true_img)
        true_img = cv2.resize(true_img,(512,512),interpolation=cv2.INTER_CUBIC)
#        cv2.imwrite("001.jpg", true_img)
        test_images = true_img.astype(np.float)
        #convert from [0:255] => [0.0:1.0]
        test_images = np.multiply(test_images, 1.0 / 255.0)
        unet2d = unet2dModule(512, 512, 1)
       # print("The predictvalue is started ")
        predictvalue = unet2d.prediction("model/unet2dglandceil.pd",test_images)
       # print("The Result is saved ")
        cv2.imwrite(writeRoot_Single, predictvalue)
   #轮廓检测 
#        gt_im2, gt_contours, gt_hierarchy = cv2.findContours(predictvalue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#        cv2.drawContours(true_img, gt_contours, -1, (0, 0, 255), 1)
#       # print("The Result is saved ")
#        cv2.imwrite(writeRoot_Toatl, true_img)
        num=num+1
        tf.reset_default_graph()#清除当前默认图中堆栈，重置默认图，实现模型参数的多次读取
        
def main(argv):
    if argv == 1:
        train()
    if argv == 2:
        predict()

if __name__ == "__main__":
   main(1)
    #main(1)
