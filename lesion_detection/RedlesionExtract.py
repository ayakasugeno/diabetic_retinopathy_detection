"""
Extract red lesion candidates from opening oparated images
※please set corresponding paths at the beginning in line 17,18
"""
import cv2, glob, os
import numpy as np
import matplotlib.pyplot as plt

###image save function
def imchange(chimg, save_path, file_name, img, i):
        cv2.imwrite(str(save_path) + str(file_name[i]), chimg)
        print(f"{img.shape} -> {chimg.shape} ")

###main function
def _main():
    ##preparation, setting paths
    get_path = "/APTOS2019 data/Enhance&Mask/opening/"    #set folders containing opening operated images
    save_path = "/APTOS2019 data/Enhance&Mask/red_lesions/"    #set folders to save output

    ##read the image file name
    os.chdir(get_path)    #move current directory
    file_name = glob.glob('*.png')

    ##load each image and extract red lesion candidates
    for i in range(len(file_name)):
        img = cv2.imread(file_name[i])    #read BGR
        chimg2 = np.zeros((416, 416))
        chimg2 = np.where(img[:, :, 1]>100, 0, 255)    
        imchange(chimg2, save_path, file_name, img, i)

if __name__ == '__main__':
    _main()