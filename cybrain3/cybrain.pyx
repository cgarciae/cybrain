from libc.stdlib cimport malloc, free
cimport libc.math as cmath
from libcpp.vector cimport vector
from cython.view cimport array
import numpy as np
cimport numpy as np
import random as rn
import math

cdef class Connection (object):

    cdef:
        public Layer source
        public Layer receiver

    def __init__(self):
        pass

    cpdef np.ndarray Zc (self):
        pass

    cpdef np.ndarray H (self):
        pass

cdef class Layer (object):

    cdef:
        public list sourceConnections
        public list receiverConnections

    def __init__(self):
        self.sourceConnections = []
        self.receiverConnections = []

    cpdef np.ndarray H (self, np.ndarray Z):
        pass

    cpdef np.ndarray Y (self):
        pass


cdef class FullConnection (Connection):

    cdef:
        public np.ndarray W
        public np.ndarray dW

    def __init__(self, int inputNeurons, int outputNeurons):
        Connection.__init__(self)

        self.W = np.empty ((inputNeurons, outputNeurons), dtype='double')
        self.dW = np.empty ((inputNeurons, outputNeurons), dtype='double')


    cpdef np.ndarray Zc (self):
        print "6"

        print self.source.Y()
        print self.W

        return self.source.Y() .dot (self.W)


cdef class LinearLayer (Layer):

    cdef:
        public np.ndarray _Y
        public bint active
        public bint back_active

    def __init__(self, int neurons):
        Layer.__init__(self)

        self._Y = np.empty ((neurons,), dtype='double')
        self.active = False
        self.back_active = False


    cpdef np.ndarray H (self, np.ndarray Z):
        return Z

    cpdef np.ndarray Y (self):
        cdef:
            np.ndarray Z
            Connection connection

        print "2"

        print self._Y.shape[0]

        if not self.active:
            self.active = True
            Z = np.zeros ((self._Y.shape[0],), dtype="double")

            print "4"

            for connection in self.sourceConnections:
                Z += connection.Zc()
                print Z

            print "5"

            self._Y = self.H (Z)

        return self._Y