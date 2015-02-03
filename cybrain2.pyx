from libc.stdlib cimport malloc, free
cimport libc.math as cmath
from libcpp.vector cimport vector
from cython.view cimport array
import random as rn
import math

cdef public int neuronCount = 0

cdef class Neuron2 (object):

    cdef:
        public double _z
        public double y
        public double dEdy
        public double dEdz
        public bint active, input

        public list forwardConnections
        public list backwardConnections

    def __init__(self):
        self._z = 0
        self.y = 0
        self.dEdy = 0
        self.dEdz = 0
        self.active = False
        self.input = True

        self.forwardConnections = []
        self.backwardConnections = []

    property z:
        def __get__(self):
            cdef Connection2 connection
            if not self.active and not self.input:
                self.active = True
                self._z = sum([connection.value for connection in self.backwardConnections])

            return self._z





cdef class Connection2 (object):

    cdef:
        public int id;
        public Neuron2 source
        public Neuron2 receiver
        double * _w
        double * _dw

    def __init__(self, Neuron2 source, Neuron2 receiver, double weight = 0):

        self._w  = <double *> malloc (sizeof (double *))
        self._dw = <double *> malloc (sizeof (double *))

        self.source = source
        self.receiver = receiver
        self._w[0] = weight
        self._dw[0] = 0

        source .forwardConnections .append (self)
        receiver .backwardConnections .append (self)

    property value:
        def __get__(self):
            return self.source.y * self.w

    property w:
        def __get__(self):
            return self._w[0]

        def __set__(self, double value):
            self._w[0] = value

    cpdef disconnect (self):
        self.source.forwardConnections.remove (self)
        self.receiver.backwardConnections.remove (self)
        self.source = None
        self.receiver = None

cdef class Layer2 (object):

    cdef:
        public list neurons
        public list forwardLayers
        public list backwardLayers
        public bint active

    def __init__(self, int neuron_number, neuronType = Neuron2):

        self.forwardLayers = []
        self.backwardLayers = []
        self.neurons = []
        self.active = False

        for _ in range (neuron_number):
            self.neurons.append (neuronType ())

    cpdef signalForwardActivation (self):
        cdef:
            Layer2 layer

        for layer in self.backwardLayers:
            layer.activate()

    cdef activationFunction (self):
        cdef:
            Neuron2 neuron

        for neuron in self.neurons:
            neuron.y = neuron.z

    cpdef activate (self):
        if not self.active:
            self.active = True

            self.signalForwardActivation()
            self.activationFunction()

    cpdef fullConnectionTo (self, Layer2 receiver):
        cdef:
            Neuron2 neuronSource
            Neuron2 neuronReceiver

        self.forwardLayers.append (receiver)
        receiver.backwardLayers.append (self)

        for neuronSource in self.neurons:
            for neuronReceiver in receiver.neurons:
                Connection2 (neuronSource, neuronReceiver)

    cpdef linearConnectionTo (self, Layer2 receiver):
        cdef:
            Neuron2 neuronSource
            Neuron2 neuronReceiver

        self.forwardLayers.append (receiver)
        receiver.backwardLayers.append (self)

        for (neuronSource, neuronReceiver) in zip (self.neurons, receiver.neurons):
            Connection2 (neuronSource, neuronReceiver)

    cpdef disconnectFrom (self, Layer2 receiver):
        cdef:
            Neuron2 neuronSource
            Connection2 connection

        self.forwardLayers.remove (receiver)
        receiver.backwardLayers.remove (self)

        for neuronSource in self.neurons:
            for connection in neuronSource.forwardConnections:
                if connection.receiver in receiver.neurons:
                    connection.disconnect()




cdef class InputLayer2 (Layer2):

    def __init__(self, int neuron_number):
        Layer2.__init__(self, neuron_number, InputNeuron2)


    cdef activationFunction (self):
        cdef InputNeuron2 neuron
        for neuron in self.neurons:
            neuron.y = neuron.x

    cpdef setData(self, double[:] data):
        cdef:
            InputNeuron2 neuron
            int i = 0
            int lengthNeurons

        lengthNeurons = len(self.neurons)

        if lengthNeurons != len(data):
            raise IndexError("data and layer dimensions are not equal")

        for i in range (lengthNeurons):
            neuron = self.neurons[i]
            neuron.x = data[i]

cdef class Net2 (object):

    cdef:
        public list inputLayers
        public list hiddenLayers
        public list outputLayers
        public list constantInput

    def __init__(self):
        self.inputLayers = []
        self.hiddenLayers = []
        self.outputLayers = []
