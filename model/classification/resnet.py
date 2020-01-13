import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from keras import layers
from keras import optimizers
from keras import models
from keras import initializers

#resnet class
class Resnet():
    def __init__(self, num_ch, input_shape, num_classes):
        self.num_ch = num_ch
        self.input_shape = input_shape
        self.num_classes = num_classes

    def model(self):
        input_shape = self.input_shape
        num_ch = self.num_ch
        num_classes = self.num_classes

        #input
        img_input = layers.Input(shape = input_shape)

        #block1
        x = layers.Conv2D(num_ch, (7,7), strides = (2,2), padding = 'same', kernel_initializer=initializers.random_normal(stddev=0.01), name = 'block1_Conv2D')(img_input)
        x = layers.MaxPooling2D((3,3), strides = (2,2), name = 'block1_maxpoolint')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        #block2,3
        x = self.residual_block(x, num_ch, 'block2')
        x = self.residual_block(x, num_ch, "block3")

        #block4,5
        x = self.residual_block_first(x, num_ch * 2, "block4")
        x = self.residual_block(x, num_ch * 2, "block5")

        #block6,7
        x = self.residual_block_first(x, num_ch * 4, "block6")
        x = self.residual_block(x, num_ch * 4, "block7")

        #block8,9
        #x = self.residual_block_first(x, num_ch * 8, "block8")
        #x = self.residual_block(x, num_ch * 8, "block9")

        #flatten
        x = self.fc(x)

        #model
        model = models.Model(img_input, x)

        return (model)


    def residual_block_first(self,input_layer, num_ch, name):
        x = layers.Conv2D(num_ch, (3,3), padding = 'same', kernel_initializer=initializers.random_normal(stddev=0.01), name = (name + "_Conv2D_1"))(input_layer)
        shortcut = layers.MaxPooling2D((2,2), name = (name + "_maxpooling"))(x)
        x = layers.Conv2D(num_ch, (3,3), padding = 'same',  kernel_initializer=initializers.random_normal(stddev=0.01), name = (name + "_Conv2D_2"))(shortcut)
        x = layers.add([x, shortcut])
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        return (x)

    def residual_block(self, input_layer, num_ch, name):
        x = layers.Conv2D(num_ch, (3,3), padding = 'same', kernel_initializer=initializers.random_normal(stddev=0.01), name = (name + "Conv2D_1"))(input_layer)
        x = layers.Conv2D(num_ch, (3,3), padding = 'same', kernel_initializer=initializers.random_normal(stddev=0.01), name = (name + "Conv2D_2"))(x)
        x = layers.add([x, input_layer])
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        return (x)

    def fc(self, input_layer, name = "Flatten"):
        x = layers.Flatten(name = name)(input_layer)
        x = layers.Dense(1024, kernel_initializer=initializers.random_normal(stddev=0.01), name = (name + "_1"))(x)
        x = layers.Dense(num_classes, kernel_initializer=initializers.random_normal(stddev=0.01), activation = 'softmax', name = (name + "_prediction"))(x)

        return (x)


#model
model = Resnet(num_ch, input_shape, num_classes).model()

#adam
adam = optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

#compile
model.compile(loss = 'categorical_crossentropy',
             optimizer = adam, metrics = ['accuracy'])

#learning
history = model.fit(train_x, train_y, validation_split=0.2, epochs = epochs, batch_size = batch_size)
