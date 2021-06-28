"""
Visualize class activation map with Grad CAM++, Faster Score CAM, and Smooth Grad using tf-keras-vis
Output all visualization map with original image in a row
※please select model used for visualization, images and true class at the beginning
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import tensorflow as tf
from tensorflow.keras.models import Model, load_model, model_from_json
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow.keras.activations as activations
import efficientnet.tfkeras as efn    
from PIL import Image

#reference: https://github.com/keisen/tf-keras-vis/blob/master/examples/attentions.ipynb
from tf_keras_vis.scorecam import ScoreCAM
from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import num_of_gpus, normalize

def _main():
    _, gpus = num_of_gpus()
    print(f'{gpus} GPUs found')

    ##experiment setting
    weights ='/content/drive/My Drive/EfficientNet/log/EfnB3_APTOS2019BlurDuplicateOmit+expandFundus_aug-Compose_2ndtrained_weights.h5'
    jsonfile = '/content/drive/My Drive/EfficientNet/EfficientNetTransferModel_Jan.7.json'    #set the model architecture with jsonfile
    image_path = '/content/drive/My Drive/APTOS2019/Blur&Duplicate_Omit/expandFundus/evaluate/0/f09cfc6a4dbd.png'
    class_index = 0    #※set the index of classes depending on the input image class 

    #load model architecture and trained weights
    json_content = open(jsonfile).read()
    model = model_from_json(json_content)
    model.load_weights(weights)

    #load image
    img = load_img(image_path, target_size=(416, 416))

    #preparing input data
    X = preprocess_input(np.array(img))
    X = tf.cast(X, tf.float32)

    #predict class by built model
    predict = model.predict(x=np.array([np.array(img)])) #predict and return the probability for each label
    pred_class = np.argmax(predict, axis=1) #return the label which have max probability value


    ##output all visualization map and original image 
    #input image
    fig = plt.figure(figsize = (20, 20))
    plt.subplot(1,4,1)
    plt.title('Input_predicted as label ' + str(pred_class))
    plt.imshow(img)

    #Get GradCAM++
    heatmap = GetGradCAMPlusPlus(class_index, X, model)
    plt.subplot(1,4,2)
    plt.title('GradCAM++')
    plt.imshow(heatmap)

    #Get Faster ScoreCAM
    heatmap = GetFasterScoreCAM(class_index, X, model)
    plt.subplot(1,4,3)
    plt.title('Faster ScoreCAM')
    plt.imshow(heatmap)

    #Get SmoothGrad
    heatmap = GetSmoothGrad(class_index, X, model)
    plt.subplot(1,4,4)
    plt.title('Smooth Grad')
    plt.imshow(heatmap)
    fig.savefig('/content/drive/My Drive/CAMoutput_' + str(class_index) + '_byExp' + str(os.path.basename(weights)) + str(os.path.basename(image_path)))

###Build calculation function
###GradCAM++
def GetGradCAMPlusPlus(class_index, img, model):
    def loss(output):
        return (output[0][class_index])
    def model_modifier(m):
        m.layers[-1].activation = tf.keras.activations.linear
        return m
    gradcam = GradcamPlusPlus(model, model_modifier= model_modifier, clone =False)
    cam = gradcam(loss, img, penultimate_layer = 'top_conv')    
    cam = normalize(cam)
    heatmap = np.uint8(cm.jet(cam[0])[..., :3]*255)
    return heatmap

###Faster ScoreCAM
def GetFasterScoreCAM(class_index, img, model):
    def loss(output):
        return (output[0][class_index])
    def model_modifier(m):
        m.layers[-1].activation = tf.keras.activations.linear
        return m
    scorecam = ScoreCAM(model, model_modifier= model_modifier, clone =False)
    cam = scorecam(loss, img, penultimate_layer = 'top_conv', max_N =10)    
    cam = normalize(cam)
    heatmap = np.uint8(cm.jet(cam[0])[..., :3]*255)
    return heatmap

###SmoothGrad
def GetSmoothGrad(class_index, img, model):
    def loss(output):
        return (output[0][class_index])
    def model_modifier(m):
        m.layers[-1].activation = tf.keras.activations.linear
        return m
    saliency = Saliency(model, model_modifier= model_modifier, clone =False)
    cam = saliency(loss, img, smooth_samples=20, smooth_noise = 0.05)    #smooth_noise: noise spread level, default =0.2. smooth_samples: the number of calculating gradient, default = 20
    cam = normalize(cam)
    heatmap = np.uint8(cm.jet(cam[0])[..., :3]*255)
    return heatmap

def loss(output, class_index):
    return (output[0][class_index])

if __name__ == '__main__':
    _main()
