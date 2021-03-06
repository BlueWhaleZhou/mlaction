#K Nearest Neighbors: KNN

import numpy as np
import operator
from os import listdir


def createdataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [1, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(x, dataset, labels, k):
    dataset_size = dataset.shape[0]
    diff_mat = np.tile(x, (dataset_size, 1)) - dataset
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
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file2matrix(filename):
    preference_dict = {'largeDoses': 3, 'smallDoses': 2, 'didntLike': 1}
    fr = open(filename)
    array_lines = fr.readlines()
    number_of_lines = len(array_lines)
    # initialize a matrix with size: number of lines x 3
    return_mat = np.zeros((number_of_lines, 3))
    class_label_vector = []
    index = 0
    for line in array_lines:
        # delete blank spaces at the beginning and the end of each line
        line = line.strip()
        # split data values according to \t
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        if list_from_line[-1].isdigit():
            class_label_vector.append(int(list_from_line[-1]))
        else:
            class_label_vector.append(preference_dict.get(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


'''
 figure plotting:
    import matplot
    import matplot.pyplot as plt

    data_matrix, labels = kNN.file2matrix(filename)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_matrix[:, 1], data_matrix[:, 2], 15.0*np.array(labels), 15.0*np.array(labels))
    plt.show()
'''


def autonorm(dataset):
    min_vals = np.amin(dataset, axis=0)
    max_vals = np.amax(dataset, axis=0)
    ranges = max_vals - min_vals
    length = dataset.shape[0]
    norm_dataset = dataset - np.tile(min_vals, (length, 1))
    norm_dataset = norm_dataset / np.tile(ranges, (length, 1))
    return norm_dataset, ranges, min_vals


def dating_class_test():
    ho_ratio = 0.10
    dating_matrix, dating_labels = file2matrix('datingTestSet.txt')
    norm_matrix, ranges, min_vals = autonorm(dating_matrix)
    length = norm_matrix.shape[0]
    num_test_vector = int(length * ho_ratio)
    error_count = 0
    for i in range(num_test_vector):
        classifier_result = classify0(norm_matrix[i, :], norm_matrix[num_test_vector:length, :], dating_labels[num_test_vector:length], 5)
        print "The testing result is: %d, the correct answer is: %d" % (classifier_result, dating_labels[i])
        if (classifier_result != dating_labels[i]):
            error_count += 1.0
    print 'The total number of testing error is : %d' % (error_count)
    print 'Error rate is : %f' % (error_count / float(num_test_vector))


def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(raw_input("percentage of time spent playing video games?"))
    ff_miles = float(raw_input("frequent flier miles earned per year?"))
    ice_cream = float(raw_input("liters of ice cream consumed per year?"))
    dating_matrix, dating_labels = file2matrix('datingTestSet.txt')
    norm_matrix, ranges, min_vals = autonorm(dating_matrix)
    input_array = np.array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((input_array - min_vals) / ranges, norm_matrix, dating_labels, 5)
    print "You will probably like this persor:", result_list[classifier_result - 1]
    # 'result - 1' since preference_dict from file2matrix function
