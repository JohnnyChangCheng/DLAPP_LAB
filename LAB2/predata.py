#!/usr/bin/python
import cv2
import os
import warnings
import math
import shutil
import cgan
import argparse


class Predata:
    def __init__(self, pathAnno, pathImg, pathTrain, pathTest):
        self.pathAnno = pathAnno
        self.pathImg = pathImg
        self.pathTrain = pathTrain
        self.pathTest = pathTest
        if (os.path.isdir(self.pathTrain)):
            warnings.warn("Folder exists recreate!", Warning)
            shutil.rmtree(self.pathTrain)
        os.mkdir(self.pathTrain)
        if (os.path.isdir(self.pathTest)):
            warnings.warn("Folder exists recreate!", Warning)
            shutil.rmtree(self.pathTest)
        os.mkdir(self.pathTest)

    def preprocess(self):
        data_cnt = 0
        file_amount = len(os.listdir(self.pathImg))
        for fnameImg in os.listdir(self.pathImg):
            fnameImgNoext = os.path.splitext(fnameImg)[0]
            number_classes = self.__split_img_class(fnameImgNoext)
            if(data_cnt > 0.8 * file_amount):
                pathOut = self.pathTest
            else:
                data_cnt = data_cnt + 1
                pathOut = self.pathTrain
            self.__crop_img_by_label(
                self.pathImg+"/"+fnameImg, number_classes, pathOut)

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
    def __crop_img_by_label(self, fnameImg, labels, pathOut):
        img = cv2.imread(fnameImg)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        srcH, srcW = img.shape[:2]
        count = 0
        fnameImgNoext = os.path.splitext(os.path.split(fnameImg)[-1])[0]
        for label in labels:
            poly_x = [(float(label[1])-float(label[3])/2) * srcW,
                      (float(label[1])+float(label[3])/2) * srcW]
            poly_y = [(float(label[2])-float(label[4])/2) * srcH,
                      (float(label[2])+float(label[4])/2) * srcH]
            crop_img = img[math.floor(poly_y[0]):math.ceil(
                poly_y[1]), math.floor(poly_x[0]):math.ceil(poly_x[1])]
            crop_img = cv2.resize(crop_img, (128, 128))
            class_dir = os.path.join(pathOut, label[0])
            if not os.path.isdir(class_dir):
                os.mkdir(class_dir)
            img_dir = os.path.join(
                class_dir, fnameImgNoext+"_" + str(count)+".jpg")
            cv2.imwrite(img_dir, crop_img)
            count = count + 1

    def sort_data_from_CGAN(self, pathCGAN, nrow):
        for fnameImg in os.listdir(pathCGAN):
            fnameImgNoext = os.path.splitext(fnameImg)[0]
            fnameImg = os.path.join(pathCGAN, fnameImg)
            img = cv2.imread(fnameImg)
            x = 0
            for label in range(0, 10):
                class_dir = os.path.join(self.pathTrain, str(label))
                x = x+2
                y = 0
                for row in range(0, nrow):
                    y = y+2
                    crop_img = img[y:y+128, x:x+128]
                    y = y+128
                    img_dir = os.path.join(
                        class_dir, fnameImgNoext+"_" + str(row)+".jpg")
                    cv2.imwrite(img_dir, crop_img)
                x = x+128

def prepare_data():
    predata = Predata("anno", "img", "dataset", "testset")
    predata.preprocess()
    cgan.train_CGAN()
    predata.sort_data_from_CGAN("images", 10)

if __name__ == '__main__':
    prepare_data()
