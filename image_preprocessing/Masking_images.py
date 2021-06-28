"""
Merge mask images with original retina images
Change background color as you want
#Please change the corresponding paths at the beginning
"""
import cv2, glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics, math

###main function
def _main():
    ##prepare
    row_image_path = "/expandFundus/enhance/"
    mask_image ="/expandFundus/mask/" 
    save_path = "/expandFundus/enhance+mask/"

    ##load retina images and corresponding mask images
    os.chdir(row_image_path)    
    file_name = glob.glob('*.png')
    for i in range(len(file_name)):
        img = cv2.imread(file_name[i])    
        mask = cv2.imread(str(mask_image) + file_name[i], cv2.IMREAD_GRAYSCALE)    
        #mask by mean RGB
        bmean = img[mask!=0].T[0].flatten().mean()
        gmean = img[mask!=0].T[1].flatten().mean()
        rmean = img[mask!=0].T[2].flatten().mean()
        img[mask==0] = [bmean, gmean, rmean]    

        cv2.imwrite(str(save_path) + str(file_name[i]), img)    #save masking image
        print(f"Mask images by mean for {file_name[i]}")

if __name__ == '__main__':
    _main()