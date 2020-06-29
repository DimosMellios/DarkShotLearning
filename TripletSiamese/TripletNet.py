#IMPORTS
from tensorflow.keras.layers import Activation,Input,Lambda, Dense, Dropout,Conv2D, MaxPooling2D, BatchNormalization,Flatten
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers


# In[ ]:


def EmbeddingNet(dims):
    """
    Base network to be shared.
    """
    model = Sequential()
    model.add(Conv2D(100,(7,7),padding='same',input_shape=(dims[0],dims[1],dims[2]),
                     activation='relu',name='conv1'))
    model.add(MaxPooling2D((2,2),(2,2),padding='same',name='pool1'))
    model.add(Conv2D(150,(5,5),padding='same',activation='relu',name='conv2'))
    model.add(MaxPooling2D((2,2),(2,2),padding='same',name='pool2'))
    model.add(Conv2D(220,(5,5),padding='same',activation='relu',name='conv3'))
    model.add(MaxPooling2D((2,2),(2,2),padding='same',name='pool3'))
    model.add(Conv2D(380,(5,5),padding='same',activation='relu',name='conv4'))
    model.add(MaxPooling2D((2,2),(2,2),padding='same',name='pool4'))
    model.add(Conv2D(460,(4,4),padding='same',activation='relu',name='conv5'))
    model.add(MaxPooling2D((2,2),(2,2),padding='same',name='pool5'))
    model.add(Dropout(0.2))
    model.add(Flatten(name='flatten'))
    model.add(Dense(512 , activation='relu',name='512-Dense')),
    model.add(Dense(128, activation = 'relu', kernel_regularizer=regularizers.l2(3e-4),
                                              bias_regularizer=regularizers.l2(3e-4),
                                              activity_regularizer=regularizers.l2(3e-5),
                                              name='128-embeddings'))
    #print(EmbeddingNet.summary())
    return model

