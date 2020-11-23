# IMPORTS
import pandas as pd
import numpy as np
import glob
import random
import cv2
from sklearn.utils import shuffle
from sklearn import preprocessing


def counter(data, location):
    """Creates a dataframe with the location and
    the category + sub-category information"""

    df = pd.DataFrame(data)
    df.columns = ['Path']
    df['Category'] = df['Sub-Category'] = ''

    # Split in categories and sub-categories
    for i in range(len(df)):
        tmp_categories = df['Path'][i].split(location)[1].split("\\")[1:3]
        df['Category'].iat[i] = tmp_categories[0]
        df['Sub-Category'].iat[i] = tmp_categories[1]

    return df


def DataLoad(location):  # , size

    # find the size of our dataset
    data = glob.glob(location + '\\***\\**\\*.png')

    # create the dataframe with the categories and shuffle the data
    info = counter(data, location)
    info = shuffle(info)

    # Encode the Labels of the dataset
    le = preprocessing.LabelEncoder()
    info['Label'] = le.fit_transform(list(info['Sub-Category']))

    # Labels to array
    labels = np.array(info['Label'])

    images = []
    # Append the images to X
    for i in info['Path']:
        image = cv2.imread(i)
        images.append(image)
    # Change the list to numpy array
    images = np.asarray(images)

    return images, labels


def make_pairs(location, split_size, pair_num):
    """Creates balanced positive and negative pairs for each image in the given dataset

    Inputs:
        The split size (tuple) for the test set
        The number of pairs (integer)"""

    x, y = DataLoad(location)
    # x = x.astype('float32')
    # x /= 255

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        for _ in range(pair_num):
            # add a matching example
            x1 = x[idx1]
            label1 = y[idx1]
            idx2 = random.choice(digit_indices[label1])
            x2 = x[idx2]

            pairs += [[x1, x2]]
            labels += [1]

            # add a not matching example
            label2 = random.randint(0, num_classes - 1)
            while label2 == label1:
                label2 = random.randint(0, num_classes - 1)

            idx2 = random.choice(digit_indices[label2])
            x2 = x[idx2]

            pairs += [[x1, x2]]
            labels += [0]

    pairs = np.array(pairs)
    labels = np.array(labels)
    # create the size of the test dataset
    test_size = int(len(labels) * (split_size))
    index = random.sample(range(len(labels)), test_size)

    # Random test sample
    pairs_test = pairs[index]
    labels_test = labels[index]

    # Remove the test samples from the train dataset
    pairs = np.delete(pairs, index, axis=0)
    labels = np.delete(labels, index)

    return pairs, labels, pairs_test, labels_test
