'''
Created on Jan 17, 2014

@author: Cristian
'''
from libc.stdlib cimport malloc, free


#TODO: class LayerConnection, class FullConnection(LayerConnections), Layer.getConnections, flag propagate for fowardInput, backwardErrorDerivative, etc.

from numpy import ndarray
from numpy cimport ndarray

cdef class Neuron(object):
    cdef:
        public list incoming_connections
        public list outgoing_connections
        public int incoming_connection_count
        public int outgoing_connection_count
        public int forward_counter
        public int backward_counter
        public float weighted_sum
        public float activation_state
        public float error_diff
        public float local_gradient
        public Layer layer       
    
    def __init__(self):
        
        self.outgoing_connections = []
        self.incoming_connections = []
        
        self.incoming_connection_count = 0
        self.outgoing_connection_count = 0
        
        
        
        self.forward_counter = 0
        self.backward_counter = 0
        
        self.weighted_sum = 0.0
        self.activation_state = 0.0
        self.error_diff = 0.0
        self.local_gradient = 0.0
        
    def addIncomingConnection(self, Connection connection):
        self.incoming_connections.append(connection)
        self.incoming_connection_count += 1
        
    def addOutgoingConnection(self, Connection connection):
        self.outgoing_connections.append(connection)
        self.outgoing_connection_count += 1
        
        
    cdef public receiveSignal(self, float signal):
        self.weighted_sum += signal
        self.forward_counter += 1

        if self.incoming_connection_count < self.forward_counter:
            raise TypeError('Cycle Detected: forward_counter = {}, incoming_connection_count = {}'.format(self.forward_counter,self.incoming_connection_count))

        if self.incoming_connection_count == self.forward_counter:
            self.propagateForward()

        
    cdef float activationFunction(self, float weighted_sum ):
        return weighted_sum
            
    cpdef propagateForward(self):
        self.activation_state = self.activationFunction(self.weighted_sum)
        
        cdef Connection connection
        for connection in self.outgoing_connections:
            connection.receiveSignal( self.activation_state )
            
    cpdef int backwardErrorSignal(self, float some_local_gradient ):
        self.error_diff += some_local_gradient
        self.backward_counter += 1
        if self.outgoing_connection_count < self.backward_counter:
            raise ValueError('Cycle Detected')
        if self.outgoing_connection_count == self.backward_counter:
            self.propagateErrorBackwards()
        return 0
        
        
    cdef float activationFunctionDerivative(self, float weighted_sum ):
        """
        dy/dz where => y = f(z)
        """
        return 1.0
    
    cpdef int propagateErrorBackwards(self):
        self.local_gradient = self.error_diff * self.activationFunctionDerivative(self.weighted_sum)
        
        cdef Connection connection
        for connection in self.incoming_connections:
            connection.backwardErrorSignal( self.local_gradient )
        return 0
            
    cpdef float outputError(self, float target):
        return 0.5*( target - self.activation_state )**2
    
    cpdef int calculateErrorDerivative(self, float target):
        self.error_diff = self.activation_state - target
        return 0
        
    
    def clearAcumulators(self):
        self.error_diff = 0.0
        self.weighted_sum = 0.0
        
    def clearCounters(self):
        self.forward_counter = 0
        self.backward_counter = 0
        
    def __repr__(self):
        return "{}".format(self.activation_state)   
    
        
cdef class LinearNeuron(Neuron):
    
    def __init__(self):
        Neuron.__init__(self)
        
        
import random as rn

cdef class Connection(object):
    """
    Connection base class
    """
    cdef:
        public Neuron source
        public Neuron destination
        float * _weight
        float * _weight_diff
    
    
    def __init__(self, Neuron source, Neuron destination, float weight = 0.0):
        self._weight = <float *> malloc( sizeof(float *) )
        self._weight_diff = <float *> malloc( sizeof(float *) )

        self.source = source
        source.addOutgoingConnection( self )
        
        self.destination = destination
        destination.addIncomingConnection( self )
        
        self._weight[0] = weight if weight else rn.uniform(-1,1);
        self._weight_diff[0] = 0.0

    property weight:
        def __get__(self):
            return self._weight[0]

        def __set__(self, value):
            self._weight[0] = value

    property weight_diff:
        def __get__(self):
            return self._weight_diff[0]

        def __set__(self, float value):
            self._weight_diff[0] = value
        
    cdef public int receiveSignal(self, float signal):
        self.fowardPropagation( self._weight[0] * signal )
        return 0
        
    def fowardPropagation(self, float signal):
        self.destination.receiveSignal( signal )
        
    def backwardErrorSignal( self, float local_gradient ):
        self._weight_diff[0] += local_gradient * self.source.activation_state
        self.backwardErrorPropagation( local_gradient )
    
    def backwardErrorPropagation(self, float local_gradient):
        self.source.backwardErrorSignal( local_gradient )
        
    def clearAcumulators(self):
        self._weight_diff[0] = 0.0
        
    def __repr__(self):
        return "source: {}, destination: {}, weight: {}".format(self.source, self.destination, self._weight[0])
        
class LinearConnection(Connection):
    
    def __init__(self, Neuron source, Neuron destination, float weight = 0.0):
        Connection.__init__(self, source, destination, weight)
        



cdef class Layer(object):
    cdef:
        public list neurons

    def __init__(self, int number_of_neurons = 0, neuron_type = LinearNeuron ):
        self.neurons = list()

        cdef int i
        for i in range(number_of_neurons):
            self.neurons.append(neuron_type())

    cpdef clearCounters(self):
        cdef Neuron neuron

        for neuron in self.neurons:
            neuron.clearCounters()

    cpdef clearAcumulators(self):
        cdef Neuron neuron

        for neuron in self.neurons:
            neuron.clearAcumulators()


    cpdef addNeurons( self, list neurons ):
        cdef Neuron neuron
        for neuron in neurons:
            self.neurons.append(neuron)
            neuron.layer = self
        
    cpdef addNeuron( self, Neuron neuron ):
        self.neurons.append( neuron )
        neuron.layer = self
        
    cpdef fowardPropagation(self):
        cdef:
            Neuron neuron
            
        for neuron in self.neurons:
            neuron.propagateForward()
            
    cpdef propagateInput(self, list input_list):
        cdef:
            Neuron neuron
            int i, N
            
        N = len(input_list)
        if len(self.neurons) != N:
            raise ValueError("target list and neuron list don't have the same lenght")
        
        for i in range(N):
            neuron = self.neurons[i]
            neuron.weighted_sum = input_list[i]
            neuron.propagateForward()
            
    cpdef backwardErrorPropagation(self):
        cdef:
            Neuron neuron
            
        for neuron in self.neurons:
            neuron.propagateErrorBackwards()
            
    cpdef calculateErrorDerivative(self, list target):
        cdef:
            Neuron neuron
            int i, N
            
        N = len(target)
        if len(self.neurons) != N:
            raise ValueError("target list and neuron list don't have the same lenght")
        
        for i in range(N):
            self.neurons[i].calculateErrorDerivative(target[i])
            
    cpdef propagateErrorDerivative(self, list targets):
        cdef:
            Neuron neuron
            int i, N
            float target
            
        N = len(targets)
        if len(self.neurons) != N:
            raise ValueError("target list and neuron list don't have the same lenght")
        
        for i in range(N):
            neuron = self.neurons[i]
            target = targets[i]
            neuron.calculateErrorDerivative(target)
            neuron.propagateErrorBackwards()
        
        
    def __repr__(self):
        cdef: 
            Neuron neuron
            
        s = ""
        
        for neuron in self.neurons:
            s += neuron.__repr__() + ", "
            
        return s

cpdef list fullConnection(Layer a, Layer b):

    cdef:
        list connection_list = list()
        list neuron_connexions
        Neuron na, nb

    for na in a.neurons:
        neuron_connexions = list()
        for nb in b.neurons:
            neuron_connexions.append(Connection(na,nb))

        connection_list.append(neuron_connexions)

    return connection_list

    
    
    
    
    