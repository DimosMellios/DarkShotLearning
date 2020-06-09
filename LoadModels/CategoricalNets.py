#IMPORTS
from tensorflow.keras.layers import Activation,Input,Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow as tf


def categorical1(image_shape, categories):
    ############# OPTION 1 #################
    ### Best accuracy -- Overfitting
    convnet1= Sequential([
        Conv2D(30,3, input_shape = image_shape),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(30,3),
        Activation('tanh'),
        MaxPooling2D(),
        Conv2D(60,2),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(120,2),
        Activation('relu'),
        Dropout(.25),
        Flatten(),
        Dense(256),
        Dense(categories , 
          kernel_regularizer=regularizers.l2(1e-4),
          bias_regularizer=regularizers.l2(1e-4),
          activity_regularizer=regularizers.l2(1e-5)),
        Activation('softmax')
    ])


    optimizer1 = Adam(0.001, decay=2.5e-4)
    convnet1.compile(loss="categorical_crossentropy", optimizer=optimizer1, metrics=['accuracy'])

    return convnet1



def categorical2(image_shape, categories):
    ############### 2nd Categorical ###################
    ### Very BAD results
    convnet2 = Sequential([
        Conv2D(32,5, input_shape = image_shape),
        Activation('tanh'),
        MaxPooling2D(2),
        Conv2D(64,5),
        Activation('relu'),
        MaxPooling2D(2),
        Conv2D(128,5),
        Activation('tanh'),
        MaxPooling2D(2),
        Conv2D(256,4),
        Activation('relu'),
        Dropout(.25),
        Flatten(),
        Dense(256),
        Activation('tanh'),
        Dense(categories , 
          kernel_regularizer=regularizers.l2(1e-4),
          bias_regularizer=regularizers.l2(1e-4),
          activity_regularizer=regularizers.l2(1e-5)),
        Activation('softmax')
    ])


    optimizer2 = RMSprop(0.001, decay=2.5e-4)
    convnet2.compile(loss="categorical_crossentropy",optimizer=optimizer2,metrics=['accuracy'])
    
    return convnet2

