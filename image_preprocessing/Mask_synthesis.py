"""
Make mask images to cover the outline of retina images
#the code automatically get image shape from data
#Please change the image path and save folder at the beginning
"""
import cv2, glob, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics, math

###main function
def _main():
    ##set paths 
    get_path = "/expandFundus/classified/healthy/"
    save_path = "/expandFundus/mask/"

    ##load retina images
    os.chdir(get_path)    
    file_name = glob.glob('*.png')
    for i in range(len(file_name)):
        img = cv2.imread(file_name[i])    #read BGR
        mask = np.zeros((img.shape[0], img.shape[1]))    #automatically set the size of image
    ###use threshold for narrowing areas to be masked
        mask = np.where(((img[:,:, 0]==img[:, :, 1])&(img[:,:, 1]==img[:,:, 2])&(img[:,:, 0] < 100)) | ((img[:,:,0]<18)&(img[:,:,1]<18)&(img[:,:,2] <18)), 0, 255)
        mask = np.float32(mask)
        noiseoutmask = cv2.medianBlur(mask, ksize=7)    #remove noise in the mask 1st time

    ###use contours for getting precise mask
        maskcopy = noiseoutmask.copy()        
        maskcopy = np.uint8(maskcopy)
        label, contour, _ = cv2.findContours(maskcopy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    #get contours
        for ii in range(len(contour)):    #fill in areas not to be masked using contours information
            if _[0][ii][3] == -1:
                cv2.drawContours(maskcopy, contour, ii, 255, -1)    
        cv2.imwrite(str(save_path) + str(file_name[i]), maskcopy)    #save masking image
        print(f"Make mask images for {file_name[i]}")

if __name__ == '__main__':
    _main()