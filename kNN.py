#K Nearest Neighbors: KNN

import numpy as np
import operator
from os import listdir

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [1, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataset, labels, k):
    dataset_size = dataset.shape[0]
    diff_mat = np.tile(inX, (dataset_size, 1)) - dataSet
    square_diff_mat = diff_mat ** 2
    square_distances = square_diff_mat.sum(axis=1)
    distances = square_distances ** 0.5
    sorted_dist_indices = np.argsort(distances)
    class_count = {}
    for i in range(k):
        vote_lable = labels[sorted_dist_indices[i]]
# count the number of occurrence for each class, if new class appears, the starting number becomes 0
        class_count[vote_lable] = class_count.get(vote_lable, 0) + 1
# return the number of classes of tuples and sort by second element
# (number of occurrence of each class in descending order)
    sortedclasscount = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedclasscount[0][0]

def file2matrix(filename):
    fr = open(filename)
    numberoflines = len(fr.readlines())
    # initialize a matrix with size: number of lines x 3
    return_mat = np.zeros((numberoflines, 3))
    class_label_vector = []
    index = 0
    for line in fr.readlines():
        '''
        line = line.strip()
        list = line.split('\t')
        return_mat[index, :] = list[0:3]
        class_label_vector.append(list[-1])
        '''
        # delete blank spaces at the beginning and the end of each line
        line = line.strip()
        # split data values according to \t
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector



