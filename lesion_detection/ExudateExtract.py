"""
Extract exudates candidates from closing oparated images
※please set corresponding paths in line 20,21
※please set corresponding image size in line 30
"""

import cv2, glob, os
import numpy as np
import matplotlib.pyplot as plt
import argparse    #for get argument parameter

###color change and save function
def imchange(chimg, save_path, file_name, img, i):
        cv2.imwrite(str(save_path) + str(file_name[i]), chimg)
        print(f"{img.shape} -> {chimg.shape} ")

###main function
def _main():
    ##set paths
    get_path = "/2ndMask/closing/"    #set folder path containing closing operated images
    save_path = "/2ndMask/closing/exudate/"    #set folder path to save output candidates

    ##read the image file name
    os.chdir(get_path)    #move current directory
    file_name = glob.glob('*.jpg')

    ##load images and extract exudate candidates
    for i in range(len(file_name)):
        img = cv2.imread(file_name[i])    #read BGR
        chimg2 = np.zeros((416, 416))    #set the image size
        chimg2 = np.where(img[:, :, 1]<180, 0, 255)
        imchange(chimg2, save_path, file_name, img, i)

if __name__ == '__main__':
    _main()