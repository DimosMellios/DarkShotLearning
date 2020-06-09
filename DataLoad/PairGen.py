# IMPORTS
import pandas as pd
import os
import numpy as np
import glob
import cv2
import random
from sklearn import preprocessing
from sklearn.utils import shuffle


def counter(data,location):
    
    '''Creates a dataframe with the location and
    the category + sub-category information'''
    
    df = pd.DataFrame(data)
    df.columns = ['Path']
    df['Category'] = df['Sub-Category'] = ''

    ## Split in categories and sub-categories
    for i in range(len(df)):

        tmp_categories = df['Path'][i].split(location)[1].split("\\")[1:3]
        df['Category'].iat[i] = tmp_categories[0]
        df['Sub-Category'].iat[i] = tmp_categories[1]
    
    return df


def DataLoad(location,split_size=None, pairs=5):
    
    '''Creates the pairs of the data and stores them in a numpy
    Inputs:
           - The directory of the final dataset
           - The split size (float) : 
                  if the user does not define the size --> no split is made (for the Testing Data)
           - The pairs is set to 5 : if your ram allows it you can generate more pairs    
           
    Returns:
           - The 2 sets of images/numpies for training and testing  (images_a, images_b, img_test_a, img_test_b)
           - The labels for the 2 sets  (y_true, y_test)
           - And the actual sets (sets)
           '''
    
    
    # find the size of our dataset
    data = glob.glob(location+'\\***\\**\\*.png')
    
    # create the dataframe with the categories and shuffle the data
    info = counter(data,location)
    info = shuffle(info)
    
    # define the size of train and test 
    x_size = int(len(info) * (1 - split_size))
    y_size = 1 - x_size
    
    # Encode the Labels of the dataset
    le = preprocessing.LabelEncoder()
    info['Label'] = le.fit_transform(list(info['Sub-Category'])) 
    
    ## Labels to array
    labels = np.array(info['Label'])
    
    images = []
    # Append the images to X
    for i in info['Path']:
        image = cv2.imread(i)
        images.append(image)
    # Change the list to numpy array
    images = np.asarray(images)   
    
    images_a = []
    images_b = []
    y_true = []
    
    sets = []   # This list is for monitoring the image_a and image_b combinations - train set
    test_sets = []  # This list is for monitoring the image_a and image_b combinations - test set
    
    # Create the dataset based on the pairs the user indicates
    ################## Training Set #########################
    
    for i in range(x_size):
        for _ in range(pairs):
            
            img2_index = i
            
            # In case the image that is picked is the same as the image_a , randomly pick another
            while img2_index == i:
                img2_index = random.randint(0,x_size-1)
                
            images_a.append(images[i])
            images_b.append(images[img2_index])
            sets.append([int(labels[i]),int(labels[img2_index])])
            
            # assign the appropriate label to the pairs
            if labels[i] == labels[img2_index]:
                # assign 1 if it is from the same category
                y_true.append(1.)
                
            else:
                # assign 0 if  the category is different
                y_true.append(0.)
    
    
    img_test_a = []
    img_test_b = []
    y_test = []
    
    #################### Testing Set #########################
    
    for j in range(x_size, len(labels)):
        for _ in range(pairs):
            
            img2_index = j
            
            # In case the image that is picked is the same as the image_a , randomly pick another
            while img2_index == j:
                img2_index = random.randint(x_size,len(images)-1)
                
            img_test_a.append(images[j])
            img_test_b.append(images[img2_index])
            test_sets.append([int(labels[j]),int(labels[img2_index])])
            
            # assign the appropriate label to the pairs
            if labels[j] == labels[img2_index]:
                # assign 1 if it is from the same category
                y_test.append(1.)
                
            else:
                # assign 0 if  the category is different
                y_test.append(0.)
      
    ##########################################################
    images_a = np.squeeze(np.array(images_a))
    images_b = np.squeeze(np.array(images_b))
    y_true   = np.squeeze(np.array(y_true))
    
    
    img_test_a = np.squeeze(np.array(img_test_a))
    img_test_b = np.squeeze(np.array(img_test_b))
    y_test   = np.squeeze(np.array(y_test))
    
    print('The initial dataset was ', len(data), ' images')
    
    
    if len(img_test_a) == 0:
        # In case the split size is 0 : just for the testing part
        
        print('\nWith 5 pairs the final dataset is consisted by:', len(y_true))
        return images_a, images_b, y_true, sets
    
    else:
        
        print('\nWith 5 pairs the final dataset is consisted by:',
              '\n-- {} Training samples'.format(len(y_true)), 
              '\n-- {} Testing samples'.format(len(y_test)))
        
        return images_a, images_b, y_true ,sets, img_test_a, img_test_b, y_test, test_sets
