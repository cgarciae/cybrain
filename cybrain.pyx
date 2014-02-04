'''
Created on Jan 17, 2014

@author: Cristian
'''
from libc.stdlib cimport malloc, free
cimport libc.math as cmath
from libcpp.vector cimport vector

#TODO: class LayerConnection, class FullConnection(LayerConnections), Layer.getConnections, flag propagate for fowardInput, backwardErrorDerivative, etc.

cdef class Neuron(object):
    cdef:
        public list incoming_connections
        public list outgoing_connections
        public int incoming_connection_count
        public int outgoing_connection_count
        public int forward_counter
        public int backward_counter
        public float weighted_sum
        public float state
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
        self.state = 0.0
        self.error_diff = 0.0
        self.local_gradient = 0.0
        
    def addIncomingConnection(self, Connection connection):
        self.incoming_connections.append(connection)
        self.incoming_connection_count += 1
        
    def addOutgoingConnection(self, Connection connection):
        self.outgoing_connections.append(connection)
        self.outgoing_connection_count += 1
        
        
    cdef public int receiveSignal(self, float signal) except -1:
        self.weighted_sum += signal
        self.forward_counter += 1

        if self.incoming_connection_count < self.forward_counter:
            raise TypeError('Cycle Detected: forward_counter = {}, incoming_connection_count = {}'.format(self.forward_counter,self.incoming_connection_count))

        if self.incoming_connection_count == self.forward_counter:
            self.propagateForward()

        
    cdef float activationFunction(self, float weighted_sum ):
        return weighted_sum
            
    cpdef propagateForward(self):
        self.state = self.activationFunction(self.weighted_sum)
        
        cdef Connection connection
        for connection in self.outgoing_connections:
            connection.receiveSignal( self.state )
            
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
        return 0.5*( target - self.state )**2
    
    cpdef errorDerivative(self, float target):
        self.error_diff = self.state - target
        
    
    def clearAcumulators(self):
        self.error_diff = 0.0
        self.weighted_sum = 0.0
        
    def clearCounters(self):
        self.forward_counter = 0
        self.backward_counter = 0
        
    def __repr__(self):
        return "{}".format(self.state)


cdef class BiasUnit(Neuron):

    def __init__(self):
        Neuron.__init__(self)
        self.state = 1.0

    cdef public int receiveSignal(self, float signal) except -1:
        pass


    cdef float activationFunction(self, float weighted_sum ):
        return 1.0

    cpdef propagateForward(self):
        cdef Connection connection
        for connection in self.outgoing_connections:
            connection.receiveSignal( self.state )

    cpdef int backwardErrorSignal(self, float some_local_gradient ):
        pass


    cdef float activationFunctionDerivative(self, float weighted_sum ):
        """
        dy/dz where => y = f(z)
        """
        return 0.0

    cpdef int propagateErrorBackwards(self):
        pass

        
cdef class LinearNeuron(Neuron):
    
    def __init__(self):
        Neuron.__init__(self)

cdef class LogisticNeuron(Neuron):

    def __init__(self):
        Neuron.__init__(self)

    cdef float activationFunction(self, float weighted_sum ):
        return 1.0 / ( 1.0 + cmath.exp(-weighted_sum))

    cdef float activationFunctionDerivative(self, float weighted_sum ):
        return self.state * (1.0 - self.state)

    cpdef float outputError(self, float target ):
        return -cmath.log( self.state ) if target == 1.0 else -cmath.log( 1.0 -self.state )

    cpdef errorDerivative(self, float target):
        self.error_diff = ( self.state - target )/( self.state * (1.0-self.state) )







        
        
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
        
    cpdef fowardPropagation(self, float signal):
        self.destination.receiveSignal( signal )
        
    cpdef backwardErrorSignal( self, float local_gradient ):
        self._weight_diff[0] += local_gradient * self.source.state
        self.backwardErrorPropagation( local_gradient )
    
    cpdef backwardErrorPropagation(self, float local_gradient):
        self.source.backwardErrorSignal( local_gradient )
        
    cpdef clearAcumulators(self):
        self._weight_diff[0] = 0.0
        
    def __repr__(self):
        return str(self._weight[0])
        
cdef class LinearConnection(Connection):
    
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
            self.neurons[i].errorDerivative(target[i])
            
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
            neuron.errorDerivative(target)
            neuron.propagateErrorBackwards()
        
        
    def __repr__(self):
        cdef: 
            Neuron neuron
            
        s = ""
        
        for neuron in self.neurons:
            s += neuron.__repr__() + ", "
            
        return s

cdef class ConnectionContainer(object):

    cdef:
        list connections


cpdef list fullConnection(Layer a, Layer b, connection = LinearConnection ):

    cdef:
        list connection_list = list()
        list neuron_connexions
        Neuron na, nb

    for na in a.neurons:
        neuron_connexions = list()
        for nb in b.neurons:
            neuron_connexions.append( connection(na,nb) )

        connection_list.append(neuron_connexions)

    return connection_list

    
    

cdef class Network(object):

    cdef:
        public list input_neurons
        public list output_neurons
        public list auto_inputs
        public list fake_outputs
        public list neurons
        public list connections

    def __init__(self):
        self.input_neurons = list()
        self.output_neurons = list()
        self.auto_inputs = list()
        self.fake_outputs = list()
        self.neurons = list()
        self.connections = list()

    cpdef addLayer(self, Layer layer):
        cdef Neuron neuron
        for neuron in layer.neurons:
            self.neurons.append(neuron)

    cpdef addConnections(self, list connections ):
        cdef:
            list neuron_connections
            Connection connection

        for neuron_connections in connections:
            for connection in neuron_connections:
                self.connections.append(connection)


    cpdef addInputLayer(self, Layer layer):
        cdef Neuron neuron
        for neuron in layer.neurons:
            self.neurons.append(neuron)
            self.input_neurons.append(neuron)

    cpdef addOutputLayer(self, Layer layer):
        cdef Neuron neuron
        for neuron in layer.neurons:
            self.neurons.append(neuron)
            self.output_neurons.append(neuron)

    cpdef addAutoInputLayer(self, Layer layer):
        cdef Neuron neuron
        for neuron in layer.neurons:
            self.neurons.append(neuron)
            self.auto_inputs.append(neuron)

    cpdef addAutoInputNeuron(self, Neuron neuron):
        self.neurons.append(neuron)
        self.auto_inputs.append(neuron)


    cpdef addFakeOutputLayer(self, Layer layer):
        cdef Neuron neuron
        for neuron in layer.neurons:
            self.neurons.append(neuron)
            self.fake_outputs.append(neuron)

    cpdef addFakeOutputNeuron(self, Neuron neuron):
        self.neurons.append(neuron)
        self.fake_outputs.append(neuron)

    cpdef list activate(self, double[:] input_data ):
        cdef:
            Neuron neuron
            int i, N
            list output_list

        N = len(input_data)
        if N != len(self.input_neurons):
            raise ValueError("Input data has wrong dimensions")


        for neuron in self.auto_inputs:
            neuron.propagateForward()

        for i in range(N):
            neuron = self.input_neurons[i]
            neuron.weighted_sum = input_data[i]
            neuron.propagateForward()

        output_list = list()
        for neuron in self.output_neurons:
            output_list.append(neuron.state)

        return output_list



    cpdef float backpropagateError(self, double[:] target_data):
        cdef:
            Neuron neuron
            int i, N
            float error

        N = len(target_data)
        if len(self.output_neurons) != N:
            raise ValueError("Target list and neuron list don't have the same lenght")

        error = 0.0
        for i in range(N):
            neuron = self.output_neurons[i]
            neuron.errorDerivative(target_data[i])
            error += neuron.outputError(target_data[i])
            neuron.propagateErrorBackwards()

        return error

    cpdef clearNetwork(self):
        cdef:
            Neuron neuron
            Connection connection

        for neuron in self.neurons:
            neuron.clearAcumulators()
            neuron.clearCounters()

        for connection in self.connections:
            connection.clearAcumulators()

    cpdef clearNeurons(self):
        cdef:
            Neuron neuron
            Connection connection

        for neuron in self.neurons:
            neuron.clearAcumulators()
            neuron.clearCounters()

    cpdef clearConnections(self):
        cdef:
            Connection connection

        for connection in self.connections:
            connection.clearAcumulators()

    

cdef class Trainer(object):

    cdef:
        vector[float*] _weights
        vector[float*] _gradient
        public float cost
        public Network net
        public double[:,:] input_data
        public double[:,:] output_data
        public float learning_rate

    def __init__(self, Network net, double[:,:] input_data, double[:,:] output_data, float learning_rate):
        cdef:
            Connection connection
            int i = 0
        self.net = net
        self.input_data = input_data
        self.output_data = output_data
        self.learning_rate = learning_rate

        for connection in net.connections:
            self._weights.push_back(connection._weight)
            self._gradient.push_back(connection._weight_diff)

    cpdef epochs(self, int epochs):
        cdef:
            int i
        for i in range(epochs):
            self.fullBatch()

    cdef fullBatch(self):
        cdef:
            int i, length_data, num_connections

        #These next values could be pre-calculated since they are constant
        length_data = len(self.input_data)
        num_connections = len(self.net.connections)

        for i in range(length_data):
            self.net.activate(self.input_data[i] )
            self.net.backpropagateError( self.output_data[i] )
            self.net.clearNeurons()

        for i in range(num_connections):
            #print connection.weight, " => ",
            self._weights[i][0] -= self.learning_rate * self._gradient[i][0]

        self.net.clearConnections()

