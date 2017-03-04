import numpy as np


def load_dataset():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_label = [0, 1, 0, 1, 0, 1] # 1: abusive, 0: not abusive
    return posting_list, class_label


def create_vocab_list(dataset):
    vocab_set = set([])
    for document in dataset:
        vocab_set = vocab_set | set(document)
    return list(vocab_set) # type(vocab_set) was set and needs to be list since set has no index


def words2vec(vocab_list, input_set):
    return_vec = [0] * len(vocab_list) # list of '0's
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1 # word exist in vocab_list: 1, otherwise: 0
        else:
            print "the word: %s is not in my vocabulary" % word
    return return_vec


def train_nb(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_c1 = sum(train_category) / float(num_train_docs)
    p_w_c1 = np.zeros(num_words)
    p_w_c0 = np.zeros(num_words)
    p0_denom = float(0)
    p1_denom = float(0)
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p_w_c1 += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p_w_c0 += train_matrix[i]
            p0_denom += sum(train_matrix[i])
    p1_vec = p_w_c1 / p1_denom
    p0_vec = p_w_c0 / p0_denom

    return p_c1, p1_vec, p0_vec









