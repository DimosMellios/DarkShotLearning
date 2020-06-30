import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy as np

def knn_baseline(x_train, y_train, x_test, y_test):
    
    dim1 = x_train.shape[1]
    dim2 = x_train.shape[2]
    dim3 = x_train.shape[3]
    
    x_train = x_train.reshape(x_train.shape[0],dim1*dim2*dim3)
    x_test = x_test.reshape(x_test.shape[0],dim1*dim2*dim3)
    
    m_iter = len(np.unique(y_test))
    k_range = range(1, m_iter)

    scores = []

    # We use a loop through the range 1 to len of classes
    # We append the scores in the dictionary
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        scores.append(metrics.accuracy_score(y_test, y_pred))



    # allow plots to appear within the notebook
    get_ipython().run_line_magic('matplotlib', 'inline')

    # plot the relationship between K and testing accuracy
    # plt.plot(x_axis, y_axis)
    plt.plot(k_range, scores)
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')

