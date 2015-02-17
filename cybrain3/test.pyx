from libc.stdlib cimport malloc, free
cimport libc.math as cmath
from libcpp.vector cimport vector
from cython.view cimport array
import numpy as np
cimport numpy as np
import random as rn
import math
import sys



ctypedef double* DoublePointer
ctypedef vector[DoublePointer] DoubleVector
ctypedef vector[DoubleVector] DoubleMatrix


cdef DoublePointer newDoublePointer ():
    return <DoublePointer> malloc (sizeof (DoublePointer))

cdef DoubleVector newDoubleVector (int columns):
    cdef:
        int j
        DoubleVector vector = (new DoubleVector(columns))[0]

    for j in range(columns):
        vector[j] = newDoublePointer()

    return vector

cdef DoubleMatrix newDoubleMatrix (int rows, int columns):
    cdef:
        int i, j
        DoubleMatrix matrix = (new DoubleMatrix(rows))[0]

    for i in range(rows):
        matrix[i] = newDoubleVector(columns)

    return  matrix

cdef DoubleMatrix m = newDoubleMatrix (3, 3)

m [1][1] = m [0][0]

m [0][0][0] = 9.9

print m [1][1][0]

cdef DoubleMatrix m2 = (new DoubleMatrix (5))[0]

m2[2] = (new DoubleVector (4))[0]
m2 [2][2] = newDoublePointer()
m2 [2][2][0] = 7.0

cdef DoublePointer p = newDoublePointer()
cdef DoublePointer p2 = newDoublePointer()

p2 = p

print p2 == p

free (p)
p = NULL


cdef void Free (DoublePointer pp) except *:
    free (pp)

try:
    Free (p2)
    Free (p)
except:
    "Unexpected error:", sys.exc_info()[0]


print m2[2][2][0]