#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#IMPORTS
from tensorflow.keras import backend as K

def triplet_loss(y_true, y_pred, margin = 0.6):
    '''Takes the tru label and the prediction label from the model
    The margin can be varied based on experimentation on each dataset'''
    
    print('y_pred.shape = ',y_pred)
    
    total_lenght = y_pred.shape.as_list()[-1]
    
    # Locate the anchor, positive and negative from the imported predictions
    anchor = y_pred[:,0:int(total_lenght*1/3)]
    positive = y_pred[:,int(total_lenght*1/3):int(total_lenght*2/3)]
    negative = y_pred[:,int(total_lenght*2/3):int(total_lenght*3/3)]

    # Euclidean distance between the anchor and the positive
    positive_dist = K.sum(K.square(anchor-positive),axis=1)

    # Euclidean distance between the anchor and the negative
    negative_dist = K.sum(K.square(anchor-negative),axis=1)

    # Calculation of the Loss based on th Euclidean distances of the positive and negative + margin
    basic_loss = K.abs(positive_dist-negative_dist+margin)
    loss = K.maximum(basic_loss,0.0)
 
    return loss

