import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

import keras
from keras import optimizers
from keras import models

from keras.layers import Input, Conv2D, Conv2DTranspose, Cropping2D, MaxPooling2D, Dropout, concatenate, Activation

class Unet():
    def __init__(self, num_ch, input_shape, num_classes):
        self.num_ch = num_ch
        self.input_shape = input_shape
        self.num_classes = num_classes

    def model(self):
        input_shape = self.input_shape
        num_ch = self.num_ch
        num_classes = self.num_classes

        #input
        img_input = Input(shape = input_shape)

        #encoding1
        encoding1 = Conv2D(num_ch, (3,3), padding = 'same', activation = 'relu', name = 'encoding1_Conv1')(img_input)
        encoding1 = Dropout(0.05)(encoding1)
        encoding1 = Conv2D(num_ch, (3,3), padding = 'same', activation = 'relu', name = 'encoding1_Conv2')(encoding1)
        encoding1_crop = Cropping2D(cropping = ((2,2), (2,2)), name = 'encoding1_crop')(encoding1)

        #encoding2
        encoding2 = MaxPooling2D((2,2), name = 'encoding2_pooling')(encoding1)
        encoding2 = Conv2D(num_ch*2, (3,3), padding = 'same', activation = 'relu', name = 'encoding2_Conv1')(encoding2)
        encoding2 = Dropout(0.05)(encoding2)
        encoding2 = Conv2D(num_ch*2, (3,3), padding = 'same', name = 'encoding2_Conv2')(encoding2)
        encoding2_crop = Cropping2D(cropping = ((1,1), (1,1)), name = 'encoding2_crop')(encoding2)

        #encoding3
        encoding3 = MaxPooling2D((2,2), name = 'encoding3_pooling')(encoding2)
        encoding3 = Conv2D(num_ch*4, (3,3), padding = 'same', activation = 'relu', name = 'encoding3_Conv1')(encoding3)
        encoding3 = Dropout(0.05)(encoding3)
        encoding3 = Conv2D(num_ch*4, (3,3), padding = 'same', name = 'encoding3_Conv2')(encoding3)
        encoding3_crop = Cropping2D(cropping = ((0,1), (0,1)), name = 'encoding3_crop')(encoding3)

        #encdoding4
        encoding4 = MaxPooling2D((2,2), name = 'encoding4_pooling')(encoding3)
        encoding4 = Conv2D(num_ch*8, (3,3), padding = 'same', activation = 'relu', name = 'encoding4_Conv1')(encoding4)
        encoding4 = Dropout(0.05)(encoding4)
        encoding4 = Conv2D(num_ch*8, (3,3), padding = 'same', name = 'encoding4_Conv2')(encoding4)

        #encdoding5
        encoding5 = MaxPooling2D((2,2), name = 'encoding5_pooling')(encoding4)
        encoding5 = Conv2D(num_ch*16, (3,3), padding = 'same', activation = 'relu', name = 'encoding5_Conv1')(encoding5)
        encoding5 = Dropout(0.05)(encoding5)
        encoding5 = Conv2D(num_ch*16, (3,3), padding = 'same', name = 'encoding5_Conv2')(encoding5)

        #decoding4
        decoding4 = Conv2DTranspose(num_ch * 8, (2,2), strides = (2,2), padding = 'same', name = 'decoding4_transpose')(encoding5)
        decoding4 = concatenate([decoding4, encoding4])
        decoding4 = Conv2D(num_ch * 8, (3,3), padding = 'same', activation = 'relu', name = 'decoding4_Conv1')(decoding4)
        decoding4 = Dropout(0.05)(decoding4)
        decoding4 = Conv2D(num_ch * 8, (3,3), padding = 'same', activation = 'relu', name = 'decoding4_Conv2')(decoding4)

        #decoding3
        decoding3 = Conv2DTranspose(num_ch * 4, (2,2), strides = (2,2), padding = 'same', name = 'decoding3_transpose')(decoding4)
        decoding3 = concatenate([decoding3, encoding3_crop])
        decoding3 = Conv2D(num_ch * 4, (3,3), padding = 'same', activation = 'relu', name = 'decoding3_Conv1')(decoding3)
        decoding3 = Dropout(0.05)(decoding3)
        decoding3 = Conv2D(num_ch * 4, (3,3), padding = 'same', activation = 'relu', name = 'decoding3_Conv2')(decoding3)

       #decoding2
        decoding2 = Conv2DTranspose(num_ch * 2, (2,2), strides = (2,2), padding = 'same', name = 'decoding2_transpose')(decoding3)
        decoding2 = concatenate([decoding2, encoding2])
        decoding2 = Conv2D(num_ch * 2, (3,3), padding = 'same', activation = 'relu', name = 'decoding2_Conv1')(decoding2)
        decoding2 = Dropout(0.05)(decoding2)
        decoding2 = Conv2D(num_ch * 2, (3,3), padding = 'same', activation = 'relu', name = 'decoding2_Conv2')(decoding2)

        #decoding1
        decoding1 = Conv2DTranspose(num_ch, (2,2), strides = (2,2), padding = 'same', name = 'decoding1_transpose')(decoding2)
        decoding1 = concatenate([decoding1, encoding1])
        decoding1 = Conv2D(num_ch, (3,3), padding = 'same', activation = 'relu', name = 'decoding1_Conv1')(decoding1)
        decoding1 = Dropout(0.05)(decoding1)
        decoding1 = Conv2D(num_ch, (3,3), padding = 'same', activation = 'relu', name = 'decoding1_Conv2')(decoding1)

        #output
        output =  Conv2D(num_classes, (1,1), activation = 'sigmoid', name = 'output_1')(decoding1)

        #model
        model = models.Model(img_input, output)

        return (model)
