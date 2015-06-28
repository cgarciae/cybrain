from libc.stdlib cimport malloc, free
cimport libc.math as cmath
from libcpp.vector cimport vector
from cython.view cimport array
import numpy as np
cimport numpy as np
import random as rn
import math
import sys

cdef double d = 0.

cdef double logisticFunctionC (double z):
    return 1. / (1. + cmath.exp (-z))

cdef double logisticFunctionP (double z):
    return 1. / (1. + math.exp (-z))

print logisticFunctionC(d)
print logisticFunctionP(d)

cdef:
    double* v
    double[:,:] A

A = np.array([[1.,2.,3.]])
v = &(A[0][1])
v[0] = 4.

print A[0][1]
