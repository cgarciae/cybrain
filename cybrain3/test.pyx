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

cdef double [:,:] dotMultiply (double [:,:] A, double[:,:] B):
    cdef:
        int i, j, k, m = A.shape[0], n = B.shape[1], o = A.shape[1]
        double acc
        double [:,:] result = array ((m, n), sizeof(double), 'd')

    print m, n

    for i in range(m):
        for j in range(n):
            acc = 0.0
            for k in range(o):
                acc += A[i,k] * B[k,j]

            result[i,j] = acc

    return result

cdef double [:,:] elementMultiply (double [:,:] A, double[:,:] B):
    cdef:
        int i, j, k, m = A.shape[0], n = A.shape[1]
        double acc
        double [:,:] result = array ((m, n), sizeof(double), 'd')

    print m, n

    for i in range(m):
        for j in range(n):
            result[i,j] = A[i,j] * B[i,j]

    return result

cdef double[:,:] elementBinaryOperation (double [:,:] A, double [:,:] B, BinaryDoubleFun f):
    cdef:
        int i, j, k, m = A.shape[0], n = A.shape[1]
        double acc
        double [:,:] result = array ((m, n), sizeof(double), 'd')

    print m, n

    for i in range(m):
        for j in range(n):
            result[i,j] = f (A[i,j], B[i,j])

    return result

cdef double[:,:] elementUnaryOperation (double [:,:] A, UnaryDoubleFun f):
    cdef:
        int i, j, k, m = A.shape[0], n = A.shape[1]
        double acc
        double [:,:] result = array ((m, n), sizeof(double), 'd')

    print m, n

    for i in range(m):
        for j in range(n):
            result[i,j] = f (A[i,j])

    return result

cdef double [:,:] A = array((2,3),sizeof(double),'d')
cdef double [:,:] B = array((3,1),sizeof(double),'d')

A [:,:] = 1.0
B [:,:] = 1.0

cdef double Double (double x):
    return 2.0 * x

cdef double Add (double x, double y):
    return x + y

print np.asarray (elementBinaryOperation (A, A, Add))
