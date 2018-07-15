import numpy as np
from math import sqrt
from collections import Counter

def KNN(k, x_train, y_train, x):
    assert 1 <= k <= x_train.shape[0], \
        '"k" is not valid ! check !'
    assert x_train.shape[0] == y_train.shape[0], \
        'Size of x_train must equal to size of y_train !'
    assert  x_train.shape[1] == x.shape[0], \
        'Feature number of x must be equal to x_train !'

    distance = [sqrt(np.sum((i-x)**2)) for i in x_train]
    neighbor_index = np.argsort(distance)
    nearest_neighbor = [y_train[i] for i in neighbor_index[:k]]

    votes = Counter(nearest_neighbor)
    return votes.most_common(1)[0][0]