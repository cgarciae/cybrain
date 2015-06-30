from libc.stdlib cimport malloc, free
cimport libc.math as cmath
from libcpp.vector cimport vector
from cython.view cimport array
import numpy as np
cimport numpy as np
import random as rn
import math
import sys

ctypedef double (*BinaryDoubleFun)(double, double)
ctypedef double (*UnaryDoubleFun)(double)

cdef double d = 0.

cdef double logisticFunctionC (double z):
    return 1. / (1. + cmath.exp (-z))

cdef double logisticFunctionP (double z):
    return 1. / (1. + math.exp (-z))

cdef double reduce (double [:] l, BinaryDoubleFun f):
    cdef:
        int i
        double elem = l[0]

    for i in range (1, l.shape[0]):
        elem = f (elem, l[i])

    return elem

print logisticFunctionC(d)
print logisticFunctionP(d)

cdef:
    double* v
    double[:,:] A

cdef double max (double a, double b):
    return a if a > b else b

cdef double min (double a, double b):
    return a if a < b else b

A = np.array([[1.,2.,3.]])
v = &(A[0][1])
v[0] = 4.

print A[0][1]

cdef double[:] B = np.array([1.,2.,-3., 6.])

print reduce (B, max)
print reduce (B, min)
