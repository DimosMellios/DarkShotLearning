#IMPORTS
from tensorflow.keras.layers import Activation,Input,Lambda, Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
import tensorflow as tf


def SiameseNet(image_shape):

    '''
    left_input = Input(image_shape)
    right_input = Input(image_shape)

    
    convnet = Sequential([
        Conv2D(5,3, input_shape=image_shape),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(5,3),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(7,2),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(7,2),
        Activation('relu'),
        Dropout(.20),
        Flatten(),
        Dense(18) , 
              kernel_regularizer=regularizers.l2(1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5)),
        Activation('sigmoid')
        ])
    '''
    
    left_input = Input(image_shape)
    right_input = Input(image_shape)

    # Embedding Net
    convnet = Sequential([
        Conv2D(50,3, input_shape=image_shape),
        Activation('relu'),
        MaxPooling2D(3),
        Conv2D(80,3),
        Activation('relu'),
        MaxPooling2D(3),
        Conv2D(80,2),
        Activation('relu'),
        MaxPooling2D(2),
        Conv2D(100,2),
        Activation('relu'),
        
        Dropout(.25),
        Flatten(),
        Dense(256),
        Dense(128 , 
              kernel_regularizer=regularizers.l2(1e-4),
              bias_regularizer=regularizers.l2(1e-4),
              activity_regularizer=regularizers.l2(1e-5)),
        
        Activation('relu') #linear
        ])
    '''

    input_shape = (120,120,3)
    convnet = Sequential([
        Conv2D(60,(5,5), input_shape=input_shape, padding = 'same'),
        Activation('relu'),
        MaxPooling2D(pool_size=(4, 4)),
        Dropout(.2),
    
        Conv2D(100, (4,4), input_shape=input_shape, padding = 'same'),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(.30),
    
    
        Conv2D(160, (3,3), input_shape=input_shape, padding = 'same'),
        Activation('relu'),
        MaxPooling2D(pool_size=(3, 3)) ,
        Dropout(.25),
    
        Conv2D(240, (7,7),input_shape=input_shape, padding = 'same'),
        Activation('relu'),
        MaxPooling2D(pool_size=(3, 3)) ,
        Dropout(.25),
    
        Flatten(),
        Dense(128, activation='sigmoid') #, activity_regularizer=regularizers.l1(1e-5)
        #Dropout(0.2)
    
        ])'''


    ##################################- 1ST OPTION- ###########################################
    
    # Connect each 'leg' of the network to each input
    # Remember, they have the same weights
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    # Getting the L1 Distance between the 2 encodings
    L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

    # Add the distance function to the network
    L1_distance = L1_layer([encoded_l, encoded_r])

    prediction = Dense(1,activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    optimizer = RMSprop(0.0001, decay=2.5e-4)
    
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])
    
    
    ###################################### --OPTION 2-- ##########################################
    
    '''
    
    def euclidean_distance(vects):
        x, y = vects
        sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
        return K.sqrt(K.maximum(sum_square, K.epsilon()))
        
    def eucl_dist_output_shape(shapes):
        shape1, shape2 = shapes
        
        return (shape1[0], 1)


    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([encoded_l, encoded_r])
    predictions = Dense(1,activation='sigmoid')(distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=predictions)


    def contrastive_loss(y_true, y_pred):
        ''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        ''
        margin = 1
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


    def accuracy(y_true, y_pred):
        ''Compute classification accuracy with a fixed threshold on distances.
        ''
        return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

    optimizer = Adam(0.0001, decay=2.5e-4)
    siamese_net.compile(loss=contrastive_loss,optimizer=optimizer,  metrics = [accuracy])
    '''
    return siamese_net

    