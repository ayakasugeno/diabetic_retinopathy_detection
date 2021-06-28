"""
Train EfficientNet pre-trained model with additional setting 
Transition of acc and loss values of 1st train and 2nd fine tuning are plotted
※please set each paths and names of experiments in line 27~32
"""

import os
import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import keras.backend as K
import tensorflow.keras.metrics
from tensorflow.keras.layers import Input, Lambda, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator    #for image pre-processing 
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model

import efficientnet.tfkeras as efn    #for efficientnet
import albumentations as albu

##need to set the path each time!!
def _main():
    ##machine learning experiment setting
    expname = "EfnB3_APTOS2019BlurDuplicateOmit+expandFundus+EnhanceMask_aug-Compose"
    train_dir =  '/content/drive/MyDrive/Dataset/APTOS2019/Blur&Duplicate_Omit/expandFundus/enhanceMask/train/'    
    log_dir = '/content/drive/My Drive/EfficientNet/log/'           
    batch_size = 16    #16 is the defult setting in this code
    num_class = 5    #set the numbers of classes. Here's 5 DR severity.
    color_fill = 179    #set the color of automatic filling in ImageDataGenerator

    ##prepare the dataset with label
    datagen = ImageDataGenerator(rotation_range=45,    
                                                                validation_split=0.2,    
                                                                fill_mode="constant",   
                                                                cval= color_fill,    #for enhancemask
                                                                preprocessing_function=wrap_compose)    #additional data augmentation

    #get data from dir path and make data batch
    train_generator = datagen.flow_from_directory(train_dir,    
                                                                                        target_size=(416, 416), color_mode = 'rgb',    
                                                                                        batch_size = batch_size, class_mode='categorical',
                                                                                        shuffle= True, subset = "training")
    num_train = train_generator.samples

    val_generator = datagen.flow_from_directory(train_dir, target_size=(416, 416), color_mode = 'rgb', 
                                                                                        batch_size = batch_size, class_mode='categorical',
                                                                                        shuffle= False, subset = "validation")
    num_val = val_generator.samples

    class_name = os.listdir(train_dir)
    print(f'Expname: {expname}')
    print(f'class name is {class_name}, class weight is {cal_weight(class_name, train_dir)}')

    ##create model from EfficientNet pre-trained model, can select from B0~B7
    pre_trained_model = efn.EfficientNetB3(input_shape = (416, 416, 3),    
                                                                include_top = False,    #Remove the last fully-connected layer at the top
                                                                weights = 'imagenet')
    for layer in pre_trained_model.layers:       #freeze the pre-trained EfficientNet model
        layer.trainable = False

    #Rebuild top
    x = GlobalAveragePooling2D(name="avg_pool")(pre_trained_model.output)
    x = Dropout(0.2, name="top_dropout")(x)    
    outputs = Dense(num_class, activation='softmax', name ="pred")(x)    #add a final layer for classification with softmax activation function

    #connect the pretrained model & rebuilt top
    model = Model(pre_trained_model.input, outputs, name = "EfficientNetTransfer")

    #set callbacks
    logging = TensorBoard(log_dir=log_dir)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-val_acc{val_categorical_accuracy:.3f}-val_loss{val_loss:.3f}.h5',    
                                                            monitor='val_loss', save_weights_only=True, save_best_only=True, period=15)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=25, verbose=1)  
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor = 0.5, patience = 6, verbose=1, mode = 'min')

    #compile
    model.compile(optimizer=Adam(lr=1e-4),
                                loss = 'categorical_crossentropy',    
                                metrics = ['categorical_accuracy'],
                                weighted_metrics=['categorical_crossentropy'])    
    ##1st train
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    History = model.fit_generator(train_generator, validation_data = val_generator,
                                                            steps_per_epoch = max(1, num_train//batch_size),
                                                            epochs = 7,
                                                            initial_epoch = 0,
                                                            validation_steps = max(1, num_val//batch_size),
                                                            class_weight = cal_weight(class_name, train_dir),
                                                            verbose = 2,
                                                            callbacks = [logging, checkpoint, early_stopping])
    ##save the model and histories and plotting
    history_df = pd.DataFrame(History.history)
    history_df.to_csv(log_dir + f"{expname}_stage1_history.csv")
    losaccplot(epoch=7,History=History,log_dir=log_dir,expname=expname)
    #announce again
    print('Finish 1st training on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    ##2nd compile & fine_tuning 
    for layer in model.layers[1:]:
        layer.trainable = True
    
    model.compile(optimizer=Adam(lr=1e-4),            # recompile to apply the change
                                loss = 'categorical_crossentropy',    
                                metrics = ['categorical_accuracy'],
                                weighted_metrics=['categorical_crossentropy'])    
    print('Unfreeze all the layers.')

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    History = model.fit_generator(train_generator, validation_data = val_generator,
                                                            steps_per_epoch = max(1, num_train//batch_size),
                                                            epochs = 200,
                                                            initial_epoch = 7,
                                                            validation_steps = max(1, num_val//batch_size),
                                                            class_weight = cal_weight(class_name, train_dir),
                                                            verbose = 2,
                                                            callbacks = [logging, checkpoint, early_stopping, reduce_lr])
    ##save the model and histories and plotting
    history_df = pd.DataFrame(History.history)
    history_df.to_csv(log_dir + f"{expname}_final_history.csv")
    model.save_weights(log_dir +f'{expname}_2ndtrained_weights.h5', save_format = 'h5')
    losaccplot(epoch=200,History=History,log_dir=log_dir,expname=expname)
    #announce again
    print('Finish 2nd training(FineTuning) on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    exp.end()

##Plot the graph of acc & loss values at selected epoch  (A.S added)
def losaccplot(epoch,History,log_dir,expname):
    fig = plt.figure(figsize=(10,5))
    #first for acc
    plt.subplot(1,2,1)
    plt.plot(History.epoch, History.history['categorical_accuracy'], label='training')
    plt.plot(History.epoch, History.history['val_categorical_accuracy'], label='validation')
    plt.title('accuracy')
    plt.xlabel('epoch')
    plt.legend()
    #second for loss
    plt.subplot(1, 2, 2)
    plt.plot(History.epoch, History.history['loss'], label='training')
    plt.plot(History.epoch, History.history['val_loss'], label='validation')
    plt.title('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    fig.savefig(log_dir + f"{expname}-Epoch{epoch}, acc loss value.png")

##Calculate the class weight for unbiased training  
##(Edited from: https://qiita.com/hiden_no_tare/items/3415b14c077cbfdadaea)
def cal_weight(class_name_list, IN_DIR):
    amounts_of_class_dict = {}
    mx = 0
    for class_name in class_name_list:
        class_dir = IN_DIR + os.sep + class_name
        file_list = os.listdir(class_dir)
        amounts_of_class_dict[class_name] = len(file_list)
        if mx < len(file_list):
            mx = len(file_list)
    class_weights = {}
    count = 0
    for class_name in class_name_list:
        class_weights[class_name_list.index(class_name)] = round(float(math.pow(amounts_of_class_dict[class_name]/mx, -1)), 2) 
    #for rearrange the class_weights dic so as not to raise error when model.fit
    class_weights2 = sorted(class_weights.items())
    class_weights.clear()
    class_weights.update(class_weights2)
    return class_weights

##Setting for implementing albumentations augmentations into keras
#good reference to decide parameters: https://albumentations-demo.herokuapp.com/
def get_augmentation():    #functions to return one of the augmentations
    train_transform = [
        albu.RandomGamma(gamma_limit=(80, 120), always_apply=False, p=0.2),    
        albu.Flip(p=0.5),
        albu.Cutout(num_holes=5, max_h_size=16, max_w_size=16, fill_value=0, p=0.5),
    ]
    return albu.Compose(train_transform)

def wrap_compose(input_image):    
    #functions to return output image tensor from input image tensor for keras implement
    transforms = get_augmentation()
    return transforms(image=input_image)["image"]

if __name__ == '__main__':
    _main()
