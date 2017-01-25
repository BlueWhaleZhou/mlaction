#K Nearest Neighbors: KNN

import numpy as np
import operator
from os import listdir

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [1, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndicies = np.argsort(distances)
    classCount = {}
    for i in range(k):
        voteIlable = labels[sortedDistIndicies[i]]
        #count the number of occurrence for each class, if new class appears, the starting number becomes 0
        classCount[voteIlable] = classCount.get(voteIlable, 0) + 1
    #return the number of classes of tuples and sort by second element(number of occurrence of each class in descending order)
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


