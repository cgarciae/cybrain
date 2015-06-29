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

    cdef vector[double*] get_gradient_pointers (self):
        pass

    cdef vector[double*] get_weight_pointers (self):
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

        ##print np.asarray (self.W)
        return dotMultiply (self.source.Y(), self.W)

    cpdef np.ndarray getW (self):
        return np.asarray (self.W)

    cpdef setW (self, np.ndarray value):
        copy_matrix(self.W, value)

    cpdef np.ndarray get_dW (self):
        return np.asarray (self.dW)

    cpdef set_dW (self, double [:,:] value):
        copy_matrix(self.dW, value)

    cdef void compute_dEdW (self):
        copy_matrix(self.dW, dotMultiply (self.source.Y().T, self.receiver.dEdZ()))

    cdef double[:,:] dEdY (self):
        self.compute_dEdW()
        return dotMultiply (self.receiver.dEdZ(), self.W.T)

    cdef vector[double*] get_gradient_pointers (self):
        return get_pointers(self.dW)

    cdef vector[double*] get_weight_pointers (self):
        return get_pointers(self.W)



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
        copy_matrix(self.dW, elementMultiply (self.source.Y(), self.receiver.dEdZ()))

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
        self.active = False
        self.back_active = False

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

    cpdef restart (self):
        self.active = False
        self.back_active = False

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
        cdef double [:,:] dEdY

        if T.shape[1] != self.neuronCount():
            raise Exception ("setTarget Error: Data shape error. Received {0}, expected {1}".format(T.shape[1], self.neuronCount()))

        dEdY = self.dEdY (T)
        self._dEdZ = elementMultiply (self.dYdZ(), dEdY)
        self.back_active = True

    cdef double[:,:] dEdY (self, double[:,:] T):
        return elementSubtract (self.Y(), T)

    cdef double[:,:] dYdZ (self):
        return np.ones ((1, self.neuronCount()), dtype="double")

    cpdef int neuronCount (self):
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

cdef class BiasUnit (LinearLayer):

    def __init__(self):
        LinearLayer.__init__(self, 1)
        self._Y[0,0] = 1.0
        self._dEdZ[0,0] = 1.0

    cdef double[:,:] H (self, double[:,:] Z):
        cdef double[:,:] h = cvarray ((1,1), sizeof(double), 'd')

        h[0,0] = 1.0

        return h


####################
##NETWORK
####################

cdef class Network (object):
    cdef:
        public list inputLayers, outputLayers, autoInputLayers, autoOutputLayers, layers, hiddenLayers, connections, all_input_layers, all_output_layers

    def __init__(self):
        self.inputLayers = []
        self.outputLayers = []
        self.autoInputLayers = []
        self.autoOutputLayers = []
        self.all_output_layers = []
        self.all_input_layers = []

        self.hiddenLayers = []
        self.layers = []
        self.connections = []

    cpdef setup (self):
        cdef:
            Layer layer

        self.all_input_layers = self.inputLayers + self.autoInputLayers
        self.all_output_layers = self.outputLayers + self.autoOutputLayers

        layers = set()
        connections = set()

        for layer in self.all_input_layers:
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

    cdef vector[double*] get_gradient_pointers (self):
        cdef:
            vector[double*] total, partial
            Connection connection

        for connection in self.connections:
            partial = connection.get_gradient_pointers()
            total.insert(total.end(), partial.begin(), partial.end())

        return total

    cdef vector[double*] get_weight_pointers (self):
        cdef:
            vector[double*] total, partial
            Connection connection

        for connection in self.connections:
            partial = connection.get_weight_pointers()
            total.insert(total.end(), partial.begin(), partial.end())

        return total

    cpdef restartNetwork (self):
        cdef Layer layer

        for layer in self.layers:
            layer.active = False
            layer.back_active = False

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

        for layer in self.all_output_layers:
            layer.Y()


    cpdef double[:,:] activate(self, double[:,:] X):
        cdef:
            Layer layer
            double[:,:] result = None

        self.restartNetwork()
        self.activate_layers(X)

        for layer in self.all_output_layers:
            if result is None:
                result = np.asarray(layer.Y())
            else:
                result = np.concatenate((result, layer.Y()), axis=1)

        return np.asarray(result)

    ## Backactivation

    cpdef setTarget (self, double[:,:] T):
        cdef:
            int start = 0, end
            LinearLayer layer

        neuron_count = sum([layer.neuronCount() for layer in self.outputLayers])
        if  neuron_count != T.shape[1]:
            raise Exception("The input data has {0} entries but there are {1} neurons.".format(T.shape[1], neuron_count))

        for layer in self.outputLayers:
            end = start + layer.neuronCount()
            layer.setTarget(T [0:1, start : end])
            start = end



    cpdef back_activate_layers(self, double[:,:] T):
        cdef:
            Layer layer


        self.setTarget(T)

        for layer in self.all_input_layers:
            layer.dEdZ()


    cpdef double[:,:] back_activate(self, double[:,:] X):
        cdef:
            Layer layer
            double[:,:] result = None

        self.restartNetwork()
        self.back_activate_layers(X)

        for layer in self.all_input_layers:
            if result is None:
                result = np.asarray(layer.dEdZ())
            else:
                result = np.concatenate((result, layer.dEdZ()), axis=1)

        return result


####################
## TRAINERS
####################
cdef class FullBatchTrainer(object):

    cdef:
        vector[double*] _weights
        vector[double*] _gradient
        double[:] total_gradient
        public double cost
        public Network net
        public double[:,:] input_data
        public double[:,:] output_data
        public double learning_rate
        public int len_gradient
        public int len_data

    def __init__(self, Network net, double[:,:] input_data, double[:,:] output_data, double learning_rate):
        self.net = net
        self.input_data = input_data
        self.output_data = output_data
        self.learning_rate = learning_rate
        self._weights = net.get_weight_pointers()
        self._gradient = net.get_gradient_pointers()
        self.len_gradient = self._gradient.size()
        self.len_data = len(input_data)
        self.len_data = len(input_data)
        self.total_gradient = cvarray ((self.len_gradient,), sizeof(double), 'd')

    cdef restart_gradient(self):
        self.total_gradient[...] = 0.0

    cdef print_gradient(self):
        cdef:
            int i
        for i in range(self.len_gradient):
            print(self.total_gradient[i])

    cdef print_actual_gradient(self):
        cdef:
            int i
        for i in range(self.len_gradient):
            print(self._gradient[i][0],)

    cdef print_weights(self):
        cdef:
            int i
        for i in range(self.len_gradient):
            print (self._weights[i][0],)

    cdef add_to_total_gradient(self):
        cdef:
            int i
        for i in range(self.len_gradient):
            self.total_gradient[i] += self._gradient[i][0]

    cpdef epochs(self, int epochs):
        cdef:
            int i
        for i in range(epochs):
            self.train()

    cdef train(self):
        cdef:
            int i
        self.restart_gradient()
        for i in range(self.len_data):
            self.net.restartNetwork()
            self.net.activate_layers(self.input_data[i:i+1,:])
            self.net.back_activate_layers(self.output_data[i:i+1,:])
            self.add_to_total_gradient()

        for i in range(self.len_gradient):
            self._weights[i][0] -= self.learning_rate * self.total_gradient[i]


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

cdef void copy_matrix (double [:,:] target, double [:,:] source):
    cdef:
        int i, j, k, m = target.shape[0], n = target.shape[1]

    if target.shape[0] != source.shape[0] or target.shape[1] != source.shape[1]:
        raise Exception("Copy error: matrices need to have the same shape")

    for i in range(m):
        for j in range(n):
            target[i,j] = source[i,j]

cdef vector[double*] get_pointers (double [:,:] A):
    cdef:
        vector[double*] result
        int i, j, m = A.shape[0], n = A.shape[1]

    for i in range(m):
        for j in range(n):
            result.push_back(&(A[i][j]))

    return result


