import theano.tensor as T
from theano import shared, function
from cython.view cimport array
import random as rn

cimport numpy as np
import numpy as np

cdef class TNeuron(object):
    cdef:
        public str name
        public list backward_connections
        public list forward_connections
        public int number_backward_connections
        public int number_forward_connections
        public object z
        public object y
        public object error_derivative
        public object local_gradient
        public TLayer layer
        public object x
        public object t
        public bint is_active
        public bint is_error_active

    def __init__(self, str name = ''):

        if name == '':
            name = str(id(self))

        self.name = name
        self.forward_connections = []
        self.backward_connections = []

        self.number_backward_connections = 0
        self.number_forward_connections = 0

        self.x = T.dscalar('x' + self.name)
        self.t = T.dscalar('t' + self.name)

        self.is_active = False
        self.is_error_active = False

    cdef addForwardConnection( self, TConnection connection ):
        self.forward_connections.append( connection )
        self.number_forward_connections += 1

    cdef addBackwardConnection( self, TConnection connection ):
        self.backward_connections.append( connection )
        self.number_backward_connections += 1

    cdef setLayer(self, TLayer layer ):
        self.layer = layer


    cpdef object activate(self):
        cdef:
            TConnection connection
            list signals = []
        if not self.is_active:
            self.is_active = True
            if self.number_backward_connections > 0:
                for connection in self.backward_connections:
                    signals.append( connection.activate() )
                self.z = T.sum(signals)
            else:
                self.z = self.x
            self.y = self.activationFuntion()
            #print "\nF => TNeuron {}, Y = {}, Z = {}, Signal = {}\n".format( self.name, self.y, self.z, signals )

        return self.y

    cpdef object errorActivate(self):
        cdef:
            TConnection connection
            list signals = []
        if not self.is_error_active:
            self.is_error_active = True
            if self.number_forward_connections > 0:
                for connection in self.forward_connections:
                    signals.append(connection.errorActivate())
                self.error_derivative = T.sum(signals)
                self.local_gradient = self.error_derivative * self.stateDerivative()
            else:
                self.local_gradient = self.errorDerivative()

            #print "\nB => TNeuron {}, Local Gradient = {}, Function Derivative = {}, Error Derivative = {}, Signal = {}\n".format( self.name, self.local_gradient, self.stateDerivative(), self.error_derivative, signals )

        return self.local_gradient

    cdef object activationFuntion(self):
        return self.z

    cdef object stateDerivative(self):
        return 1.0

    cdef object E(self):
        return 0.5*( self.t - self.y )**2

    cdef object errorDerivative(self):
        return self.y - self.t

    cdef clear(self):
        self.is_active = self.is_error_active = False

    cdef list getConnections(self):
        return self.forward_connections

    def __repr__(self):
        return str(self.y)

cdef class TLogisticNeuron(TNeuron):

    def __init__(self):
        TNeuron.__init__(self)

    cpdef object activationFuntion(self):
        return 1.0 / ( 1.0 + T.exp(-self.z ) )

    cpdef object stateDerivative(self):
        return self.y - self.y**2

    cpdef object E(self):
        return -self.t * T.log(self.y) - (1.0 - self.t) * T.log( 1.0 - self.y )

cdef class TBiasUnit(TNeuron):

    def __init__(self):
        TNeuron.__init__(self)
        self.x = 1.0
        self.y = 1.0
        self.is_active = True

    cdef object activationFuntion(self):
        return 1.0


cdef class TConnection(object):
    """
    Connection base class
    """
    cdef:
        public str name
        public TNeuron source
        public TNeuron destination
        public object weight
        public object weight_diff


    def __init__(self, TNeuron source, TNeuron destination, double weight = 0.0):

        self.name = source.name + "_" + destination.name

        self.source = source
        source.addForwardConnection( self )

        self.destination = destination
        destination.addBackwardConnection( self )

        self.weight = weight if weight else rn.uniform(-1,1)
        name = 'w' + self.name


    cdef object activate(self):
        cdef:
            object signal = self.source.activate()
            object product = signal * self.weight

        #print "    Connection {}, Weight {}, Signal {}, Product {}".format( self.name, self.weight, signal, product )
        return product

    cdef object errorActivate(self):
        cdef:
            object signal = self.destination.errorActivate()
            object product = signal * self.weight

        self.weight_diff = signal * self.source.y

        #print "    Connection {}, Product {}, Weight {}, Signal {}, Source State = {}, Weight Diff = {}".format( self.name, product, self.weight, signal, self.source.y, self.weight_diff )
        return signal * self.weight

    def __repr__(self):
        return str(self.weight)

cdef class TLayer(object):
    cdef:
        public list neurons

    def __init__(self, *args, neuron_type = TNeuron, list names = [] ):
        cdef:
            list type_list
            int i
        self.neurons = list()
        if len(args) == 2:
            type_list = args[0]*[ args[1] ]
        elif type(args[0]) is int:
            type_list = args[0] * [neuron_type]
        else:
            type_list = args[0]
        for neuron_type in type_list:
            self.neurons.append( neuron_type() )

        if names:
            for i in range(len(names)):
                self.neurons[i].name = names[i]
                self.neurons[i].x = T.dscalar('x' + self.neurons[i].name)

    def addNeuron(self, TNeuron neuron ):
        self.neurons.append( neuron )
        neuron.setLayer(self)

    cpdef removeNeuron(self, TNeuron neuron):
        self.neurons.remove(neuron)
        neuron.setLayer( None )

    cpdef list getConnections(self):
        cdef:
            list connections = list()
            TNeuron neuron
        for neuron in self.neurons:
            connections += neuron.getConnections()
        return connections

    cpdef clearNeurons(self):
        cdef:
            TNeuron neuron
        for neuron in self.neurons:
            neuron.clear()

    cdef object error(self):
        cdef:
            list error = []
            TNeuron neuron
        for neuron in self.neurons:
            error.append( neuron.E() )

        return T.sum(error)

    #TODO: activate, errorActivate

    cpdef activate(self):
        cdef:
            TNeuron neuron
        for neuron in self.neurons:
            neuron.activate()

    cpdef errorActivate(self):
        cdef:
            TNeuron neuron
        for neuron in self.neurons:
            neuron.errorActivate()


    cpdef list connectTo( self, TLayer b, connection = TConnection ):

        cdef:
            list connection_list = list()
            list neuron_connexions
            TNeuron na, nb

        for na in self.neurons:
            neuron_connexions = list()
            for nb in b.neurons:
                neuron_connexions.append( connection(na,nb) )

            connection_list.append(neuron_connexions)

        return connection_list

    def __repr__(self):
        return str([ n.__repr__() for n in self.neurons ])

    def __getitem__(self, item):
        return self.neurons[item]

cdef class TNetwork(object):

    cdef:
        public list input_layers
        public list output_layers
        public list auto_inputs
        public list fake_outputs
        public list layers
        public object weights
        public object gradient

        public object f_states
        public object f_gradient
        public object f_error
        public object f_update


    def __init__(self):
        self.input_layers = list()
        self.output_layers = list()
        self.auto_inputs = list()
        self.fake_outputs = list()
        self.layers = list()

    cpdef addLayer(self, TLayer layer):
        self.layers.append(layer)

    cpdef addInputLayer(self, TLayer layer):
        self.input_layers.append(layer)

    cpdef addOutputLayer(self, TLayer layer):
        self.output_layers.append(layer)

    cpdef addAutoInputLayer(self, TLayer layer):
        self.auto_inputs.append(layer)

    cpdef addFakeOutputLayer(self, TLayer layer):
        self.fake_outputs.append(layer)

    cpdef object getWeights(self):
        return self.weights

    cpdef sortWeights(self):
        cdef:
            TConnection connection
            list connections = self.getConnections()
            int length = len(connections)
            int i
            np.ndarray weights = np.zeros( (length,) )

        i = 0
        for connection in connections:
            weights[i] = connection.weight if type(connection.weight) == float else self.weights[i].eval()
            i += 1

        self.weights = shared( weights )

        i = 0
        for connection in connections:
            connection.weight = self.weights[i]
            i += 1

    cpdef sortGradient(self):
        self.gradient = self.getGradient()

    cpdef sort(self):
        self.sortWeights()
        self.activate()
        self.backpropagate()
        self.sortGradient()

    cpdef list getConnections(self):
        cdef:
            TLayer layer
            list connections = list()
        for layer in self.input_layers + self.layers + self.auto_inputs:
            connections += layer.getConnections()
        return connections

    cpdef object getGradient(self):
        cdef:
            TConnection connection
            list connections = self.getConnections()
            list gradient = list()
        for connection in connections:
            gradient.append(connection.weight_diff)

        return T.stacklists( gradient )

    cpdef list getOutputs(self):
        cdef:
            TLayer layer
            TNeuron neuron
            list output = list()
        for layer in self.output_layers:
            for neuron in layer.neurons:
                output.append(neuron.y)
        return output

    cpdef list getInputs(self):
        cdef:
            TLayer layer
            TNeuron neuron
            list output = list()
        for layer in self.input_layers:
            for neuron in layer.neurons:
                output.append(neuron.x)
        return output

    cpdef list getTargets(self):
        cdef:
            TLayer layer
            TNeuron neuron
            list output = list()
        for layer in self.output_layers:
            for neuron in layer.neurons:
                output.append(neuron.t)
        return output

    cpdef activate(self):
        cdef:
            TLayer layer

        self.clearLayers()
        for layer in self.output_layers + self.fake_outputs:
            layer.activate()

    cpdef backpropagate( self ):
        cdef:
            TLayer layer

        self.clearLayers()
        for layer in self.input_layers + self.auto_inputs:
            layer.errorActivate()

    cpdef clearLayers(self):
        cdef TLayer layer
        for layer in self.output_layers + self.input_layers + self.layers + self.auto_inputs + self.fake_outputs:
            layer.clearNeurons()

    cpdef object error( self ):
        cdef:
            list error = list()
            TLayer layer

        for layer in self.output_layers + self.fake_outputs:
            error.append( layer.error() )

        return T.sum( error )


    cpdef compile(self):
        self.sortWeights()
        self.activate()
        self.backpropagate()
        self.sortGradient()

        y = self.getOutputs()
        x = self.getInputs()
        t = self.getTargets()
        E = self.error()
        w = self.weights
        dw = self.gradient

        self.f_states = function( x, y )
        self.f_gradient = function( x+t , dw )
        self.f_error = function( x+t, E )
        self.f_update = function( [dw], updates= {w: dw}, on_unused_input='ignore')


    def __call__(self, *args, **kwargs):
        return self.f_states(*args)

    def num_error(self, *args):
        return self.f_error(args)

    def num_gradient(self, *args):
        return self.f_gradient(*args)

    def num_weights(self, *args):
        return self.f_weights(*args)

    def update(self, vals):
        self.f_update(vals)



cdef class TTrainer(object):

    cdef:
        list weights
        double[:] gradient
        double[:] total_gradient
        public double cost
        public TNetwork net
        public np.ndarray input_data
        public np.ndarray output_data
        public double learning_rate
        public int len_gradient
        public int len_data

    def __init__(self, TNetwork net, np.ndarray input_data, np.ndarray output_data, double learning_rate):
        cdef:
            TConnection connection
            list connections
            int i = 0
        self.net = net
        self.input_data = input_data
        self.output_data = output_data
        self.learning_rate = learning_rate
        self.len_gradient = len(net.getConnections())
        self.len_data = len(input_data)


    cpdef epochs(self, int epochs):
        cdef:
            int i
        for i in range(epochs):
            self.fullBatch()

    cdef fullBatch(self):
        cdef np.ndarray dw = np.zeros( (self.len_gradient,) )
        for i in range(len(self.input_data)):
            dw += self.net.f_gradient( *(tuple(self.input_data[i]) + tuple(self.output_data[i])) )
        self.net.update( self.net.weights.eval() - self.learning_rate * dw )
