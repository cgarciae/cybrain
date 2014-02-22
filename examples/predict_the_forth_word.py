__author__ = 'Cristian'

import scipy.io
import numpy as np
mat = scipy.io.loadmat('word_data.mat')
data = mat['data'][0][0]

test_set = (data[0]-1).T
train_set = (data[1]-1).T
valid_set = (data[2]-1).T

dictionary = data[3][0]
dictionary = np.array([ str(dictionary[i][0]) for i in range(len(dictionary)) ])

import csv

np.savetxt("dict.csv", dictionary, delimiter=",", fmt= '%s')
