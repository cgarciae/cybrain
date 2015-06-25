from libc.stdlib cimport malloc, free
cimport libc.math as cmath
from libcpp.vector cimport vector
from cython.view cimport array as cvarray
from cpython.array cimport array, clone
import numpy as np
cimport numpy as np
import random as rn
import math
import exceptions

#BUILD: python setup.py build_ext --inplace

####################
##TYPES AND TEMPS
####################

ctypedef double (*BinaryDoubleFun)(double, double)
ctypedef double (*UnaryDoubleFun)(double)

cdef array template = array ('d')

####################
##CONNECTIONS
####################

cdef class Connection (object):

    cdef:
        public Layer source
        public Layer receiver

    def __init__(self):
        pass

    cdef double[:,:] Zc (self):
        pass

    cdef double[:,:] dEdY (self):
        pass

    cdef void compute_dEdW (self):
        pass


cdef class FullConnection (Connection):

    cdef:
        double[:,:] W
        double[:,:] dW

    def __init__(self, LinearLayer source, LinearLayer receiver, double weightMagnitud = 1.0):
        Connection.__init__(self)

        self.W = weightMagnitud * (np.random.rand (source.neuronCount(), receiver.neuronCount()) * 2.0 - 1.0)
        self.dW = np.empty ((source.neuronCount(), receiver.neuronCount()), dtype='double')

        source.receiverConnections.append (self)
        receiver.sourceConnections.append (self)

        self.source = source
        self.receiver = receiver


    cdef double[:,:] Zc (self):


        if self.source.Y().shape[1] != self.W.shape[0]:
            raise Exception ("Shape Error")

        if self.receiver.neuronCount() != self.W.shape[1]:
            raise Exception ("Neuron Difference Error")

        return dotMultiply (self.source.Y(), self.W)

    cpdef np.ndarray getW (self):
        return np.asarray (self.W)

    cpdef setW (self, np.ndarray value):
        self.W = value

    cpdef np.ndarray get_dW (self):
        return np.asarray (self.dW)

    cpdef set_dW (self, double [:,:] value):
        self.dW = value

    cdef void compute_dEdW (self):
        ##dW : [n, m]
        ##(dZdW = Y) : [1, n]
        ##dEdZ.shape : [1, m]
        ##(Y.T * dEdZ) : [n, 1] * [1, m]  =  [n, m]
        self.dW = dotMultiply (self.source.Y().T, self.receiver.dEdZ())

    cdef double[:,:] dEdY (self):
        self.compute_dEdW()
        ##(dZdY = W) : [n, m]
        ##dEdZ : [1, m]
        ##dEdY : [1, n]
        ##(W * dEdZ.T).T :  ([n, m] * [m, 1] = [n, 1]).T = [1, n]
        ##(W * dEdZ.T).T = dEdZ * W.T : [1, m] * [m, n] = [1, n]
        return dotMultiply (self.receiver.dEdZ(), self.W.T)



cdef class LinearConnection (FullConnection):

    def __init__(self, LinearLayer source, LinearLayer receiver, double weightMagnitud = 1.0):
        FullConnection.__init__(self, source, receiver, weightMagnitud)

        if source.neuronCount() != receiver.neuronCount():
            raise Exception("Number of neurons not equal")

        self.W = weightMagnitud * (np.random.rand (1, receiver.neuronCount()) * 2.0 - 1.0)
        self.dW = np.empty ((1, receiver.neuronCount()), dtype='double')


    cdef double[:,:] Zc (self):
        if self.source.Y().shape[1] != self.W.shape[1]:
            raise Exception  ("Shape Error")

        if self.receiver.neuronCount() != self.W.shape[1]:
            raise Exception ("Neuron Difference Error")

        return elementMultiply (self.source.Y(), self.W)

    cdef void compute_dEdW (self):
        self.dW = elementMultiply (self.source.Y(), self.receiver.dEdZ())

    cdef double[:,:] dEdY (self):
        self.compute_dEdW()
        return elementMultiply (self.receiver.dEdZ(), self.W)

####################
##LAYERS
####################

cdef class Layer (object):

    cdef:
        public list sourceConnections
        public list receiverConnections
        public bint active
        public bint back_active

    def __init__(self):
        self.sourceConnections = []
        self.receiverConnections = []

    cdef double[:,:] H (self, double[:,:] Z):
        pass

    cdef double[:,:] Y (self):
        pass

    cdef double[:,:] dEdY (self, double[:,:] T):
        pass

    cdef double[:,:] dYdZ (self):
        pass

    cdef double[:,:] dEdZ (self):
        pass

    cdef int neuronCount (self):
        pass
    
    cpdef setData (self, double [:,:] value):
        pass

cdef class LinearLayer (Layer):

    cdef:
        public double[:,:] _Y, Z, _dEdZ, _dEdY


    def __init__(self, int neuron_number):
        Layer.__init__(self)

        self._Y = np.empty ((1, neuron_number), dtype='double')
        self.Z = np.empty ((1, neuron_number), dtype='double')
        self._dEdZ = np.empty ((1, neuron_number), dtype='double')
        self._dEdY = np.empty ((1, neuron_number), dtype='double')
        self.active = False
        self.back_active = False


    cdef double[:,:] H (self, double[:,:] Z):
        return Z

    cdef double[:,:] Y (self):
        cdef:
            Connection connection
            int i


        if not self.active:
            self.active = True
            self.Z[...] = 0.0

            for connection in self.sourceConnections:
                self.Z = elementAdd (self.Z, connection.Zc())

            self._Y = self.H (self.Z)

        return self._Y

    cdef double[:,:] dEdZ (self):
        cdef:
            Connection connection
            int i

        if not self.back_active:
            self.back_active = True
            self._dEdY[...] = 0.0

            for connection in self.receiverConnections:
                self._dEdY = elementAdd (self._dEdY, connection.dEdY())

            self._dEdZ = elementMultiply (self.dYdZ(), self._dEdY)

        return self._dEdZ

    cpdef np.ndarray get_dEdZ (self):
        return np.asarray (self.dEdZ())


    cpdef np.ndarray getY (self):
        return np.asarray (self.Y())

    cpdef setData (self, double [:,:] value):
        #TODO: check dimensions
        if value.shape[1] != self.neuronCount():
            raise Exception ("setData Error: Data shape error. Received {0}, expected {1}".format(value.shape[1], self.neuronCount()))

        self._Y = value
        self.active = True

    cpdef setTarget (self, double [:,:] T):
        cdef double [:,:] dEdY = self.dEdY (T)
        self._dEdZ = elementMultiply (self.dYdZ(), dEdY)
        self.back_active = True

    cdef double[:,:] dEdY (self, double[:,:] T):
        return elementSubtract (self.Y(), T)

    cdef double[:,:] dYdZ (self):
        return np.ones ((1, self._Y.shape[1]), dtype="double")

    cdef int neuronCount (self):
        return self._Y.shape[1]

    cpdef LinearLayer fullConnectTo (self, LinearLayer layer, double weightMagnitud = 1.0):
        FullConnection (self, layer, weightMagnitud)
        return  layer

    cpdef LinearLayer linearConnectTo (self, LinearLayer layer, double weightMagnitud = 1.0):
        LinearConnection (self, layer, weightMagnitud)
        return  layer

cdef class LogisticLayer (LinearLayer):

    cdef double[:,:] H (self, double[:,:] Z):
        return elementUnaryOperation (Z, logisticFunction)

    cdef double[:,:] dEdY (self, double[:,:] T):
        return elementBinaryOperation (T, self.Y(), logistic_dEdY)

    cdef double[:,:] dYdZ (self):
        return elementUnaryOperation(self.Y(), logistic_dYdZ)


####################
##NETWORK
####################

cdef class Network (object):
    cdef:
        public list inputLayers, outputLayers, autoInputLayers, autoOutputLayers, layers, hiddenLayers, connections

    def __init__(self):
        self.inputLayers = []
        self.outputLayers = []
        self.autoInputLayers = []
        self.autoOutputLayers = []

        self.hiddenLayers = []
        self.layers = []
        self.connections = []

    cpdef findHiddenComponents (self):
        cdef:
            Layer layer

        layers = set()
        connections = set()

        for layer in self.inputLayers + self.autoInputLayers:
            self.recursivelyFindLayersAndConnections (layer, layers, connections)

        self.layers = list (layers)
        self.connections = list (connections)


    cdef void recursivelyFindLayersAndConnections (self, Layer layer, layers, connections):
        cdef:
            Connection connection
            Layer nextLayer

        layers.add (layer)
        for connection in layer.receiverConnections:
            connections.add (connection)
            self.recursivelyFindLayersAndConnections (connection.receiver, layers, connections)

    cpdef restartNetwork (self):
        cdef Layer layer

        for layer in self.layers:
            layer.active = False

    cpdef setData (self, double[:,:] X):
        cdef:
            int start = 0, end
            LinearLayer layer

        neuron_count = sum([layer.neuronCount() for layer in self.inputLayers])
        if  neuron_count != X.shape[1]:
            raise Exception("The input data has {0} entries but there are {1} neurons.".format(X.shape[1], neuron_count))

        for layer in self.inputLayers:
            end = start + layer.neuronCount()
            layer.setData(X [0:1, start : end])
            start = end



    cpdef activate_layers(self, double[:,:] X):
        cdef:
            Layer layer

        self.setData(X)

        for layer in self.layers:
            layer.Y()


    cpdef double[:,:] activate(self, double[:,:] X):
        cdef:
            Layer layer
            double[:,:] result = None

        self.activate_layers(X)

        for layer in self.layers:
            if result is None:
                result = layer.getY()
            else:
                result = np.concatenate((result, layer.Y()), axis=1)

        return result


####################
##HELPER FUNCTIONS
####################
cdef double logistic_dEdY (double t, double y):
    return (t - y) / ((-1.0 + y) * y)

cdef double logistic_dYdZ (double y):
    return y * (1.0 - y)

cdef double quadraticDistance (double a, double b):
    return 0.5 * (a - b)**2

cdef double logisticFunction (double z):
    return 1. / (1. + cmath.exp (-z))

cdef double [:,:] dotMultiply (double [:,:] A, double[:,:] B):
    cdef:
        int i, j, k, m = A.shape[0], n = B.shape[1], o = A.shape[1]
        double acc
        double [:,:] result = cvarray ((m, n), sizeof(double), 'd')


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
        double [:,:] result = cvarray ((m, n), sizeof(double), 'd')

    for i in range(m):
        for j in range(n):
            result[i,j] = A[i,j] * B[i,j]

    return result

cdef double [:,:] elementAdd (double [:,:] A, double[:,:] B):
    cdef:
        int i, j, k, m = A.shape[0], n = A.shape[1]
        double acc
        double [:,:] result = cvarray ((m, n), sizeof(double), 'd')

    if A.shape[0] != B.shape[0] or A.shape[1] != B.shape[1]:
        raise Exception ("Dimension Error")

    for i in range(m):
        for j in range(n):
            result[i,j] = A[i,j] + B[i,j]

    return result

cdef double [:,:] elementSubtract (double [:,:] A, double[:,:] B):
    cdef:
        int i, j, k, m = A.shape[0], n = A.shape[1]
        double acc
        double [:,:] result = cvarray ((m, n), sizeof(double), 'd')

    for i in range(m):
        for j in range(n):
            result[i,j] = A[i,j] - B[i,j]

    return result

cdef double[:,:] elementBinaryOperation (double [:,:] A, double [:,:] B, BinaryDoubleFun f):
    cdef:
        int i, j, k, m = A.shape[0], n = A.shape[1]
        double acc
        double [:,:] result = cvarray ((m, n), sizeof(double), 'd')

    for i in range(m):
        for j in range(n):
            result[i,j] = f (A[i,j], B[i,j])

    return result

cdef double[:,:] elementUnaryOperation (double [:,:] A, UnaryDoubleFun f):
    cdef:
        int i, j, k, m = A.shape[0], n = A.shape[1]
        double acc
        double [:,:] result = cvarray ((m, n), sizeof(double), 'd')


    for i in range(m):
        for j in range(n):
            result[i,j] = f (A[i,j])

    return result