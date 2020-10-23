#!/usr/bin/python
import cv2
import os
import warnings
import math
import shutil

class Predata:
    def __init__(self, pathAnno, pathImg,pathOut):
        self.pathAnno = pathAnno
        self.pathImg = pathImg
        self.pathOut = pathOut

    def preprocess(self):
        if ( os.path.isdir(self.pathOut)):
            warnings.warn("Folder exists recreate!", Warning)
            shutil.rmtree(self.pathOut)
        
        os.mkdir(self.pathOut)

        for fnameImg in os.listdir(self.pathImg):
            fnameImgNoext=os.path.splitext(fnameImg)[0]
            number_classes = self.__split_img_class(fnameImgNoext)
            self.__crop_img_by_label(self.pathImg+"/"+fnameImg,number_classes)

    def __split_img_class(self, fnameImg):
        pathAnno = self.pathAnno+"/"+fnameImg+".txt"
        numclasses = []
        if(os.path.isfile(pathAnno)):
            with open(pathAnno, 'r') as fptr:
                lines = fptr.readlines()
                for line in lines:
                    numclasses.append(line.split())
        return numclasses

    #Yolo Label <object-class> <x_center> <y_center> <width> <height>
    def __crop_img_by_label(self, fnameImg, labels):
        img = cv2.imread(fnameImg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        srcH, srcW = img.shape[:2]
        count = 0
        fnameImgNoext=os.path.splitext(os.path.split(fnameImg)[-1])[0]
        for label in labels:
            poly_x = [(float(label[1])-float(label[3])/2 )* srcW, (float(label[1])+float(label[3])/2)* srcW] 
            poly_y = [(float(label[2])-float(label[4])/2 )* srcH, (float(label[2])+float(label[4])/2 )* srcH]
            crop_img = img[math.floor(poly_y[0]):math.ceil(poly_y[1]), math.floor(poly_x[0]):math.ceil(poly_x[1])]
            crop_img = cv2.resize(crop_img, (128, 128))
            class_dir = os.path.join(self.pathOut, label[0])
            print(class_dir)
            if not os.path.isdir(class_dir):
                os.mkdir(class_dir)
            img_dir = os.path.join(class_dir, fnameImgNoext+"_"+ str(count)+".jpg")
            print(img_dir)
            cv2.imwrite(img_dir, crop_img)
            count = count + 1

if __name__ == '__main__':                   
    predata = Predata("anno","img","dataset")
    predata.preprocess()