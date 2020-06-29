#IMPORTS
import pandas as pd
import numpy as np
import glob
import random
import cv2
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from itertools import permutations

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


def DataLoad(location): #, size
    
    # find the size of our dataset
    data = glob.glob(location+'\\***\\**\\*.png')
    
    # create the dataframe with the categories and shuffle the data
    info = counter(data,location)
    
    #Encode the labels
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
    
    return images, labels, info




def triplet_gen(x,y, test_size=0.3, num_pairs=10):
    
    '''Generates the triplets based on the 
    number of pairs defined by the user'''
    
    dataset = tuple([x,y])
    
    train_size = 1-test_size

    triplets_train = []
    triplets_test = []
    
    # Creates the pairs
    for data_class in sorted(set(dataset[1])):

        similar = np.where((dataset[1] == data_class))[0]
        different = np.where(dataset[1] != data_class)[0]
        
        # Create the Anchor - Positive pairs 
        Positive_pairs = random.sample(list(permutations(similar,2)),k=num_pairs) 
        # Creates the Anchor - Negative pairs
        Negative_pairs = random.sample(list(different),k=num_pairs)
        

        #Training sets
        AP = len(Positive_pairs)
#         N = len(Negative_pairs)
        for p in Positive_pairs[:int(AP*train_size)]:
            Anchor = dataset[0][p[0]]
            Positive = dataset[0][p[1]]
            for n in Negative_pairs:
                Negative = dataset[0][n]
                triplets_train.append([Anchor,Positive,Negative]) 
        
        #Testing Sets
        for p in Positive_pairs[int(AP*train_size):]:
            Anchor = dataset[0][p[0]]
            Positive = dataset[0][p[1]]
            for n in Negative_pairs:
                Negative = dataset[0][n]
                triplets_test.append([Anchor,Positive,Negative])    
                
    return np.array(triplets_train), np.array(triplets_test)

