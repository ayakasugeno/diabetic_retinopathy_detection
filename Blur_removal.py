"""
Detected blur images as poor samples and copied only fine samples in save_path

#Please change the image path: get_path and save_path corresponding to your file paths
##get_path: file path which images to be checked are contained
#Please ensure to set extension in file_name, ex. '*.png' or '*.jpg', which is corresponding to your dataset 
"""
import cv2, glob, os, sys, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statistics, math

###main function
def _main():
    ##set paths for saving and loading image files
    get_path = "D:/ImageRecognitionProject/Retinal fundus data samles/Kaggle,Diabetic_Retinopathy_Detection/raw_images/expand_fundus/"
    save_path = "D:/ImageRecognitionProject/Retinal fundus data samles/Kaggle,Diabetic_Retinopathy_Detection/raw_images/expand_fundus/blur_omit/"

    ##load images to check
    os.chdir(get_path)    
    file_name = glob.glob('*.png')

    #prepare for save the information
    namelist =[]
    colname = ['image','stdDev_of_LaplacianGrayscale', 'stdDev_of_LaplacianGrayscale_nonzero']
    namelist.append(colname)

    ##get images and calculate values from green channel
    for i in range(len(file_name)):
        img = cv2.imread(file_name[i])
        name = os.path.basename(file_name[i])
        namee = [name]

    #use Laplacian to detect edge in images and calculate the variance of Laplacian returns
        edge = cv2.Laplacian(img, cv2.CV_8U, ksize =5)
        grayscale = cv2.cvtColor(edge, cv2.COLOR_BGR2GRAY)
        stdDev = np.std(grayscale)    #return standard deviation in all areas
        stdDevnz = np.std(grayscale[np.nonzero(grayscale)])    #return standard deviation without 0 areas

    #set threshold of blurring image and save in blur folder
        if stdDevnz <55:    
            namee.append(stdDev)
            namee.append(stdDevnz)
            namelist.append(namee)

        else:    #others go to fine samples
            cv2.imwrite(str(save_path) + str(name.split('.')[0]) + '.png', img)
            shutil.copy(src = file_name[i], dst = save_path + str(name.split('.')[0]) + '.png')
            print(f'{name} checked')  #, and copied to Fine file; {save_path}')

    list = pd.DataFrame(namelist)
    list.to_csv(save_path + 'detected,blur_samples.csv')

if __name__ == '__main__':
    _main()