from libc.stdlib cimport malloc, free
cimport libc.math as cmath
from libcpp.vector cimport vector
from cython.view cimport array
import numpy as np
cimport numpy as np
import random as rn
import math

ctypedef double (*BinaryDoubleFun)(double, double)
ctypedef double (*UnaryDoubleFun)(double)


cdef class Connection (object):

    cdef:
        public Layer source
        public Layer receiver

    def __init__(self):
        pass

    cpdef double[:,:] Zc (self):
        pass

    cpdef double[:,:] H (self):
        pass

cdef class Layer (object):

    cdef:
        public list sourceConnections
        public list receiverConnections

    def __init__(self):
        self.sourceConnections = []
        self.receiverConnections = []

    cpdef double[:,:] H (self, double[:,:] Z):
        pass

    cpdef double[:,:] Y (self):
        pass


cdef class FullConnection (Connection):

    cdef:
        public double[:,:] W
        public double[:,:] dW

    def __init__(self, int inputNeurons, int outputNeurons):
        Connection.__init__(self)

        self.W = np.empty ((inputNeurons, outputNeurons), dtype='double')
        self.dW = np.empty ((inputNeurons, outputNeurons), dtype='double')


    cpdef double[:,:] Zc (self):
        cdef double[:,:] res

        print "6"

        print np.asarray (self.source.Y())
        print np.asarray (self.W)

        res = dotMultiply (self.source.Y(), self.W)

        print "aca"

        print np.asarray (res)

        return res


cdef class LinearLayer (Layer):

    cdef:
        public double[:,:] _Y
        public bint active
        public bint back_active

    def __init__(self, int neurons):
        Layer.__init__(self)

        self._Y = np.empty ((1, neurons), dtype='double')
        self.active = False
        self.back_active = False


    cpdef double[:,:] H (self, double[:,:] Z):
        return Z

    cpdef double[:,:] Y (self):
        cdef:
            double[:,:] Z, Zc
            Connection connection
            int i

        print "2"

        print self._Y.shape[0]

        if not self.active:
            self.active = True
            Z = np.zeros ((1, self._Y.shape[1]), dtype="double")

            print "4"

            for connection in self.sourceConnections:
                Zc = connection.Zc()
                for i in range(Zc.shape[1]):
                    Z[0,i] += Zc[0,i]

            print "5"

            self._Y = self.H (Z)

        return self._Y


cdef double [:,:] dotMultiply (double [:,:] A, double[:,:] B):
    cdef:
        int i, j, k, m = A.shape[0], n = B.shape[1], o = A.shape[1]
        double acc
        double [:,:] result = array ((m, n), sizeof(double), 'd')

    print A.shape[0], A.shape[1]
    print B.shape[0], B.shape[1]

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