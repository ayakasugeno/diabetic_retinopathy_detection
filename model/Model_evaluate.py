"""
Evaluate the trained models developed for 0~4 classification
※please change each paths and setting at the beginning
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import keras.backend as K
from tensorflow.keras.layers import Input, Lambda, Flatten, Dense, Dropout, GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model, Sequential, load_model, model_from_json
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator    #for image pre-processing 
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model

import efficientnet.tfkeras as efn    #for efficientnet
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import datetime

from sklearn.metrics import confusion_matrix, cohen_kappa_score, f1_score, roc_curve, auc, plot_confusion_matrix
from sklearn.preprocessing import label_binarize
from itertools import cycle

##need to set the path each time!!
def _main():
    ##machine learning experiment setting
    test_dir =  '/content/drive/My Drive/APTOS2019/Blur&Duplicate_Omit/expandFundus/SeverityClassification/evaluate/'    ##need to change
    weight ='/content/drive/My Drive/EfficientNet/log/EfnB3_APTOS2019BlurDuplicateOmit+expandFundus_aug-Compose_2ndtrained_weights.h5'
    batch_size =16
    expname = os.path.splitext(os.path.basename(weight))[0]
    evfilename = os.path.basename(test_dir)
    print(f"expname: {expname}, Images to use for evaluate: {evfilename}")
    num_class = 5

    ##prepare the test dataset with label
    datagen = ImageDataGenerator()    
    test_generator = datagen.flow_from_directory(test_dir,    # get image dataset and labeling name
                                                                                        target_size=(416, 416), color_mode = 'rgb',    
                                                                                        class_mode='categorical', shuffle = False)
    num_tes = test_generator.samples

    ##build model
    pre_trained_model = efn.EfficientNetB3(input_shape = (416, 416, 3),    # shape of the image 変更する
                                                                include_top = False,    #Remove the last fully-connected layer at the top
                                                                weights = 'imagenet')
    x = GlobalAveragePooling2D(name="avg_pool")(pre_trained_model.output)
    x = Dropout(0.2, name="top_dropout")(x)    #If set dropout before Batch Normalization, performance will get worse, so just 1 dropout layer just before softmax would be nice. https://github.com/arXivTimes/arXivTimes/issues/608
    outputs = Dense(num_class, activation='softmax', name ="pred")(x)    #add a final layer for classification with softmax activation function
    model = Model(pre_trained_model.input, outputs, name = "EfficientNetTransfer")

    #load weight
    model.load_weights(weight)
    for layer in model.layers:
        layer.trainable = False
        if isinstance(layer, BatchNormalization):
            layer.trainable = False    

    #compile before testing
    model.compile(optimizer=Adam(lr=1e-4),
                                loss = 'categorical_crossentropy',    
                                metrics = ['categorical_accuracy'],
                                weighted_metrics=['categorical_crossentropy'])    

    ###Evaluate model
    print(f'Evaluate the model on {num_tes} test images with batch size {batch_size}')
    results = model.evaluate(x= test_generator, verbose =1, batch_size = batch_size)
    print(f'{model.metrics_names}: {results}')    #results is list

    ###Evaluate model from prediction
    print(f'Predict lebels on {num_tes} test images')
    y_pred = model.predict(x=test_generator, verbose = 1, batch_size = batch_size)    #predict and return the probability for each label
    pred_class = np.argmax(y_pred, axis =1)    #return the label which have max probability value
    pred_class_ascend = np.argsort(y_pred)        #get all prediction

    #output all predictions with csv
    trulab = pd.Series(test_generator.labels, dtype='int8')
    print(f'trulab: {trulab}')
    pred_c = pd.DataFrame(pred_class_ascend, dtype='int8')
    print(f'pred_c: {pred_c}')
    output = pd.concat([trulab, pred_c], axis=1)
    output.columns = ['True labels', '5th predicted labels', '4th predicted labels', '3rd', '2nd', '1st']
    output.to_csv(f'/content/drive/My Drive/EfficientNet/{expname}-Predictions.csv')

    ##make confusion matrix
    cm = confusion_matrix(test_generator.labels, pred_class)
    print(f"Confusion matrix: {cm}")

    data = {'True label': test_generator.labels,
                    'Predict label': pred_class }
    df = pd.DataFrame(data, columns=['True label','Predict label'])
    confmtpd = pd.crosstab(df['True label'], df['Predict label'], dropna = False)
    print (f"Confusion matrix with pandas: {confmtpd}")    #confmtpd is pd.dataframe

    #plot confusin matrix
    cfmfig = plt.figure()
    sn.heatmap(confmtpd, annot=True, cmap='Greens', fmt = 'd')
    cfmfig.savefig(f'{expname}-Confusin Matrix.png')

    ##F1 score calculation
    #for each class
    eachf1 = f1_score(y_true = test_generator.labels, y_pred = pred_class, labels = [0, 1, 2, 3, 4], average = None)    
    #for averaged F1 score from all classes, weighted ver.
    avf1 = f1_score(y_true = test_generator.labels, y_pred = pred_class, labels = [0, 1, 2, 3, 4], average = 'weighted')    #can select from 'weighted', 'macro','micro' etc
    #for averaged F1 score from all classes, unweighted ver.
    avf2 = f1_score(y_true = test_generator.labels, y_pred = pred_class, labels = [0, 1, 2, 3, 4], average = 'macro')    #can select from 'weighted', 'macro','micro' etc

    print(f'f1 score of each class: {eachf1}, averaged f1_score: {avf1}, averaged f1_score, weighted ver: {avf2}')

    ##calculate kappa statistics by 2 ways, kappa score (-1~1) assess the agreement
    kappa = cohen_kappa_score(y1= test_generator.labels, y2= pred_class, labels = [0,1,2,3,4])
    weightedkappa = cohen_kappa_score(y1= test_generator.labels, y2= pred_class, labels = [0,1,2,3,4],
                                                        weights = 'linear')
    quadratickappa = cohen_kappa_score(y1= test_generator.labels, y2= pred_class, labels = [0,1,2,3,4],
                                                        weights = 'quadratic')

    ##ROC curve
    #prepare
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    thresholds = dict()
    y_true = label_binarize(test_generator.labels, classes =[0,1,2,3,4])
    n_classes = y_true.shape[1]

    #compute ROC curve for each class                      
    for i in range(n_classes):
        fpr[i], tpr[i], thresholds[i] = roc_curve(y_true[:, i], y_pred[:, i])    #input the predicted probability and true labels with 0or1 for each score
        roc_auc[i] = auc(fpr[i], tpr[i])

    #plot ROC curve
    fig = plt.figure()
    colors = cycle(['aqua', 'darkorange','cornflowerblue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                        label='Classes {0}, AUC = {1:0.2f}'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1-specificity)')
    plt.ylabel('True Positive Rate (sensitivity)')
    plt.title('ROC curve of each class')
    plt.legend(loc = "lower right")

    #save figure
    fig.savefig(f"/content/drive/My Drive/機械学習/EfficientNet/{expname}_ROC.png")

    #save all the results in excels
    newlist = []
    mod = ["Model", expname]
    EvaluateRes = [f"Test {model.metrics_names[0]}", results[0], f"Test {model.metrics_names[1]}", results[1], f"test {model.metrics_names[2]}", results[2]]
    F1scoreRes = ["F1_score of each class", eachf1, "F1_score, weighted_average", avf1, "F1_score, averaged", avf2]
    KappaRes = ["Kappa score", kappa, "Kappa score, linear weighted", weightedkappa, "Kappa score, quadratic weighted", quadratickappa]
    Thres = ["ROC curve", "Label 0", "1", "2", "3", "4"]
    Thres2 = ["Thresholds value", thresholds[0], thresholds[1], thresholds[2], thresholds[3], thresholds[4]]    

    newlist.append(mod)
    newlist.append(EvaluateRes)
    newlist.append(F1scoreRes)
    newlist.append(KappaRes)
    newlist.append(Thres)
    newlist.append(Thres2)

    summary = pd.DataFrame(newlist)
    summary.to_csv(f'/content/drive/My Drive/EfficientNet/{expname}-Evaluate-{datetime.datetime.now()}.csv')

if __name__ == '__main__':
    _main()
