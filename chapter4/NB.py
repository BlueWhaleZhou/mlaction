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
    return_vec = [0] * len(vocab_list) # list of '0'
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1 # word exist in vocab_list: 1, otherwise: 0
        else:
            print "the word: %s is not in my vocabulary" % word
    return return_vec




