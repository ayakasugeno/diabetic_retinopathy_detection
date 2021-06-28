"""
Expand fundus areas using mask and save the created images
※be careful if error returns during calculation after finding contours, this code skipped the images and will continuously run throughout target images
"""
import cv2, glob, os, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.morphology import white_tophat, closing, disk, black_tophat

def _main():
    ##set each paths
    image_path = "/APTOS2019/raw_images/"
    mask_path = "/APTOS2019/raw_images/Mask/"
    save_path = "/APTOS2019/raw_images/expand_fundus/"
    size = 416

    ##read the images from get_path
    os.chdir(image_path)    #move current directory
    file_name = glob.glob('*.jpeg')
    print(f'Number of images: {len(file_name)}')

    for i in range(len(file_name)):
        img = cv2.imread(file_name[i])
        mask = cv2.imread(mask_path + file_name[i], cv2.IMREAD_GRAYSCALE)
        ret, binmask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY+ cv2.THRESH_OTSU)

        #fundus detection by detecting concours and getting max contour area
        _, contours, hierarchy = cv2.findContours(binmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=lambda x: cv2.contourArea(x), default =-1)

        #if program find contours, expand fundas area
        if max_contour != '-1':
            #Fitting
            try:    #in Kaggle DR Detection database (or for the images previously detected as blur), error returns during calculating minEnclosingCircles.
                      #so run this program throughout the images, set try: and except:
                (cx, cy), radius = cv2.minEnclosingCircle(max_contour)

                #crop fundus areas 
                x = max(0, int(cx)-int(radius))
                y = max(0, int(cy)-int(radius))
                fundus = img[y:int(cy)+int(radius), x: int(cx)+int(radius), :]
                #expand
                scale = size/max(fundus.shape[0], fundus.shape[1])
                refundus = cv2.resize(fundus, dsize=None, fx=scale, fy=scale)

                cv2.imwrite(save_path + os.path.splitext(os.path.basename(file_name[i]))[0] + '.png', refundus)
                print(f"Expand fundus of {file_name[i]} done")

            except:
                print(f"Error in {file_name[i]}")

        #if problem could not find contours, skip the image
        else:
            print(f"Could not find contours for {file_name[i]}")

if __name__ == '__main__':
    _main()
