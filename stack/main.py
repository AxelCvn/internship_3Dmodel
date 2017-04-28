# -*- coding: utf-8 -*-
import os, sys, random
import numpy
import pylab
from PIL import Image
from math import *
import time

import package.func as func
import package.cnn as cnn
import package.all_mri as mlFunc

import progressbar

def fc(path):
    list_fc = []
    i = 0
    maxlen = len(os.listdir(path))
    with progressbar.ProgressBar(max_value=maxlen) as bar:
        for filename in os.listdir(path)  :
            filePath = os.path.join(path,filename)
            if not os.path.isdir(filePath) and filename.endswith('.png'):
                #print filePath
                fc = cnn.run_cnn(filePath)
                list_fc.append(fc)
                bar.update(i)
            i+=1

    fc_path = os.path.join(path,'fc_bis.txt')
    #print fc_path
    fc_file = open(fc_path, 'w')
    for item in list_fc:
        fc_file.write("%s\n" % item)

    return fc_path

def pickImages(stackPath):
    # Compute a number of file to rotate to be able to predict rotation
    num_files = len([f for f in os.listdir(stackPath)
            if os.path.isfile(os.path.join(stackPath, f))])
    nb_data_img = (num_files//100)
    print "Number of images to rotate : " + str(nb_data_img)

    images = []
    if nb_data_img < 10 :
        nb_data_img = 10
    i = 1
    while i <= nb_data_img :
        tmp_img = random.choice(os.listdir(stackPath))
        #print tmp_img
        if not tmp_img in images and tmp_img.endswith('.png') :
            images.append(tmp_img)
            i += 1
        else :
            print "Bad luck, already have this image"
    return images


def main():
    main_time = time.time()
    #
    # # Stack of images to process
    # stackPath = '/home/axel/Documents/DATASET/canon_T2star_vp.tif_17566'
    #
    # # Check if images are Tiff, in that case convert to PNG
    # # We assume that the directory contains only the images to process (it can also contain other directories)
    # stackPath = func.processDir(stackPath)
    #
    # # If not already done, resize images
    # stackPath = func.resize(stackPath)
    #
    # #Pick images to rotate
    # rndImg = pickImages(stackPath)
    # print rndImg
    #
    # # Rotate the images
    # rotatedTrain, rotTrainTxt = func.rotateStack(stackPath, rndImg)
    #
    # # Apply convolution on the rotated stack
    #
    # fcTrain = fc(rotatedTrain)
    #
    #
    # rotatedStack, rotValTxt = func.rotateStackVal(stackPath)
    #
    # fcVal = fc(rotatedStack)


    fcTrain = '/home/axel/Documents/DATASET/canon_T2star_vp.tif_17566/png/resized/rotated/fc_bis.txt'
    rotTrainTxt = '/home/axel/Documents/DATASET/canon_T2star_vp.tif_17566/png/resized/rotated/rotation.txt'

    fcVal = '/home/axel/Documents/DATASET/canon_T2star_vp.tif_17566/png/resized/rotatedVal/fc_bis.txt'
    rotValTxt = '/home/axel/Documents/DATASET/canon_T2star_vp.tif_17566/png/resized/rotatedVal/rotationVal.txt'

    rotatedStack = '/home/axel/Documents/DATASET/canon_T2star_vp.tif_17566/png/resized/rotatedVal'

    # pred = '/home/axel/Documents/DATASET/canon_T2star_vp.tif_17566/png/resized/rotatedVal/rfr_pred_val_0_100.csv'

    pred = mlFunc.all(fcTrain, rotTrainTxt, fcVal, rotValTxt, 1)

    # func.correctStack(rotatedStack, pred)
    print("--- %s seconds --- GLOBAL TIME" % (time.time() - main_time))

    #Apply convolution on images
    return 0

if __name__ == "__main__":
    main()
