"""
Image Processing from resized images
※Contain enhance, opening, and closing operation image processing
※Please set corresponding paths at the beginning in line 26, 27
"""
import cv2, glob, os
import numpy as np
import matplotlib.pyplot as plt
import argparse    #for get argument parameter

###set the getting arguments function
def get_args():
    parser = argparse.ArgumentParser(description="Change the color scale of input images by setting output color of images like '--color enhance', '-col enhance'")
    parser.add_argument('-col' , '--color', type=str, help="select from 'enhance','opening','closing'")
    args = parser.parse_args()
    return args.color

###color change and save function
def imchange(chimg, save_path, file_name, colset, img, i):
        cv2.imwrite(str(save_path) + str(file_name[i]), chimg)
        print(f"{img.shape} -> {chimg.shape} , {colset} ")

###main function
def _main():
    ##prepare
    get_path = "/APTOS2019 data/expandFundus/maskbymean/"    #set folder paths containing images to be processed
    save_path = "/APTOS2019 data/expandFundus/enhance/"    #set folder to save output images
    colset = get_args()

    ##read the image file name
    os.chdir(get_path)    #move current directory
    file_name = glob.glob('*.png')

    ##change color
    for i in range(len(file_name)):
        img = cv2.imread(file_name[i])
        chimg = np.empty_like(img)    #set empty data for recognizable error

        #prepare ndarray? of zero for separate 3 colors by 'blue, red, or green'
        height, width = img.shape[:2]
        channels = 1 
        zeros = np.zeros((height, width), img.dtype)

        if colset == "enhance":    #referred to https://pdfslide.net/documents/kaggle-diabetic-retinopathy-detection-competition-report.html
            scale = 416
            chimg = cv2.addWeighted(img, 4,
                                                            cv2.GaussianBlur(img, (0, 0), scale/30), -4,
                                                            128)
            imchange(chimg, save_path, file_name, colset, img, i)            

        elif colset =='opening':    #opening the images, one of the morphological transformation
            kernel = np.ones((3,3), np.uint8)
            chimg = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            imchange(chimg, save_path, file_name, colset, img, i)

        elif colset =='closing':    #closing the images, one of the morphological transformation
            kernel = np.ones((3,3), np.uint8)
            chimg = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            imchange(chimg, save_path, file_name, colset, img, i)

        else: 
            print("Error. Set color argument correctly.")
            break

if __name__ == '__main__':
    _main()