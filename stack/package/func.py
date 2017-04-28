# -*- coding: utf-8 -*-
import os, sys
import numpy
from PIL import Image
import shutil
from math import *

#Check if first file of a directory is tif or not
def processDir(stackPath) :
    for root, dirs, files in os.walk(stackPath, topdown=True):
        for name in files:
            if name.endswith('tif'):
                stackIsTif = True
                break
            else :
                stackIsTif = False
                break
        break

    #If stackIsTif convert to PNG
    if stackIsTif :
        print"Let's convert stack to PNG"
        stackPath = tifToPng(stackPath)
    else :
        print"We can now resize the stack"
    return stackPath

#Create a png copy of images of a stack and return the path of the new stack
def tifToPng(directory):
    print" Working in : " + directory + " directory"
    #Create new directory to store the new png images
    new_dir = os.path.join(directory,'png')
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    #For each image of the stack, create a png copy and save it in the new directory
    for fileName in os.listdir(directory) :
        imgPath = os.path.join(directory,fileName)
        if not os.path.isdir(imgPath) and (imgPath.endswith('.tif')):
            with Image.open(imgPath) as im:
                finalPath = os.path.join(new_dir,fileName)
                pre, ext = os.path.splitext(finalPath)
                finalPath = finalPath = pre + '.png'
                im.mode='I'
                im.point(lambda i:i*(1./256)).convert('L').save(finalPath)
        else :
            filePath = os.path.join(directory,fileName)
    return new_dir

# Resize the stack such as rotations don't crop the original image
def resize(stackPath):
    #First check if stack is already resized
    for root, dirs, files in os.walk(stackPath, topdown=True):
        for name in files:
            testPath = os.path.join(stackPath,name)
            pre, ext = os.path.splitext(testPath)
            if pre.endswith('resized'):
                resized = True
                break
            else :
                resized = False
                break
        break

    if not resized :
        print'Start resizing stack'
        #Create a new directory to store resized images
        new_dir = os.path.join(stackPath,'resized')
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)

        #Resize each image and save it in the new directory
        for fileName in os.listdir(stackPath) :
            filePath = os.path.join(stackPath,fileName)
            if not os.path.isdir(filePath) and fileName.endswith('.png') :
                with Image.open(filePath) as im:
                    if im.size[0] > im.size[1] :
                        newSize = im.size[0]
                    else :
                        newSize = im.size[1]


                    old_size = im.size
                    new_size = (newSize, newSize)

                    diago = int(round(sqrt(2)*newSize))
                    new_size = (diago,diago)

                    new_im = Image.new("L", new_size)
                    new_im.paste(im,((new_size[0]-im.size[0])/2,(new_size[1]-im.size[1])/2))

                    newFileName = os.path.join(new_dir,fileName)
                    pre, ext = os.path.splitext(newFileName)
                    newFileName = pre + '_resized' + ext
                    new_im.save(newFileName)

            else :
                print" WARNING " +str(filePath) + " does not exist or is not a png image !!"
        return new_dir
    else :
        print'This stack is already resized, we can rotate it'
        return stackPath

def rotateImg(stackPath, nb, imgName, rotatedDirectory):
    # Get image path
    imgPath = os.path.join(stackPath, imgName)

    # Open image
    with Image.open(imgPath) as im:
        rot = -90 + nb*1.8
        rotation = (numpy.random.uniform((rot-0.1),(rot+0.1),1)[0])
        rotatedImg = im.rotate(rotation)

        ext =  '_' + str(nb) +'.png'
        rotImgPath = imgName.replace('.png', ext)
        rotImgPath = os.path.join(rotatedDirectory, rotImgPath)
        rotatedImg.save(rotImgPath)
    return rotation

def rotateStack(stackPath, imgs):
    #Create rotated directory if it does not exists
    rotatedDir = os.path.join(stackPath, 'rotated')
    if os.path.exists(rotatedDir):
        for the_file in os.listdir(rotatedDir):
            file_path = os.path.join(rotatedDir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
        print os.listdir(rotatedDir)
    else :
        os.mkdir(rotatedDir)

    rotationAngle_list = []
    for img in imgs :
        print img
        i = 1
        while i <= 100 :
            rot = rotateImg(stackPath, i, img, rotatedDir)
            rotationAngle_list.append(rot)
            i += 1
    #otate_randomly_dataSet(path)
    rot_file = os.path.join(rotatedDir,'rotation.txt')
    rotFile = open(rot_file, 'w')
    rotStr = str(rotationAngle_list)
    rotStr = rotStr.replace("[","")
    rotStr = rotStr.replace("]","")
    rotFile.write(rotStr)
    return rotatedDir, rot_file

def rotateImgVal(img, imgPath, rotatedDirectory):
    rotImgPath = os.path.join(rotatedDirectory, img)
    # Open image
    with Image.open(imgPath) as im:
        rotation = numpy.random.uniform(-90,90,1)[0]
        rotatedImg = im.rotate(rotation)

        rotatedImg.save(rotImgPath)
    return rotation

def rotateStackVal(stackPath):
    #Create rotated directory if it does not exists
    rotatedDir = os.path.join(stackPath, 'rotatedVal')
    if os.path.exists(rotatedDir):
        for the_file in os.listdir(rotatedDir):
            file_path = os.path.join(rotatedDir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
        print os.listdir(rotatedDir)
    else :
        os.mkdir(rotatedDir)

    rotationAngle_list = []
    for img in os.listdir(stackPath):
        imgPath = os.path.join(stackPath, img)
        if os.path.exists(imgPath) and imgPath.endswith('.png'):
            rot = rotateImgVal(img, imgPath, rotatedDir)
            rotationAngle_list.append(rot)
    rot_file = os.path.join(rotatedDir,'rotationVal.txt')
    rotFile = open(rot_file, 'w')
    rotStr = str(rotationAngle_list)
    rotStr = rotStr.replace("[","")
    rotStr = rotStr.replace("]","")
    rotFile.write(rotStr)
    return rotatedDir, rot_file

# Apply predicted rotation to the rotated stack to correct it and save the corrected stack in a new directory
def correctStack(stackPath, rotationFile):
    correctedDir = os.path.join(stackPath, 'corrected')
    if os.path.exists(correctedDir):
        for the_file in os.listdir(correctedDir):
            file_path = os.path.join(correctedDir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path): shutil.rmtree(file_path)
            except Exception as e:
                print(e)
        print os.listdir(correctedDir)
    else :
        os.mkdir(correctedDir)

    rotFile = open(rotationFile, 'r')
    rotations = rotFile.read().split(',')
    print rotations
    for i in range(len(rotations)) :
        rotations[i] = float(rotations[i])
    files = []
    for fileName in os.listdir(stackPath) :
        filePath = os.path.join(stackPath, fileName)
        if not os.path.isdir(filePath) and filePath.endswith('.png'):
            files.append(fileName)
            print fileName

    try:
        if len(files) == len(rotations):
            print ' GO FOR CORRECTION'
            for i in range (len(files)):
                curFile = files[i]
                print curFile
                rot = -rotations[i]
                print rot
                imgPath = os.path.join(stackPath, curFile)
                print imgPath
                with Image.open(imgPath) as img :
                    newImgPath = os.path.join(correctedDir, curFile)
                    img = img.rotate(rot)
                    img.save(newImgPath)
            return correctedDir
    except Exception as e :
        print (e)
