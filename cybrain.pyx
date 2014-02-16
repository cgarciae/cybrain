from libc.stdlib cimport malloc, free
cimport libc.math as cmath
from libcpp.vector cimport vector
from cython.view cimport array
import random as rn
import math

cdef class Neuron(object):
    cdef:
        public str name
        public list backward_connections
        public list forward_connections
        public int number_backward_connections
        public int number_forward_connections
        public double z
        public double y
        public double dEdy
        public double dEdz
        public Layer layer
        public double x
        public double t
        public bint is_active_sum
        public bint is_active_state
        public bint is_error_derivative_active
        public bint is_local_gradient_active
    
    def __init__(self, str name = ''):

        if name == '':
            name = str(id(self))

        self.name = name
        self.forward_connections = []
        self.backward_connections = []
        
        self.number_backward_connections = 0
        self.number_forward_connections = 0
        
        self.z = 0.0
        self.y = 0.0
        self.dEdy = 0.0
        self.dEdz = 0.0
        self.x = 0.0
        self.t = 0.0

        self.is_active_sum = False
        self.is_active_state = False
        self.is_error_derivative_active = False
        self.is_local_gradient_active = False

    cpdef Connection connectTo(self, Neuron neuron, str name = '', weight = 0.0, connection_type = Connection ):
        cdef Connection new = connection_type( self, neuron, name= name, weight= weight )
        return new

    cdef addForwardConnection( self, Connection connection ):
        self.forward_connections.append( connection )
        self.number_forward_connections += 1

    cdef addBackwardConnection( self, Connection connection ):
        self.backward_connections.append( connection )
        self.number_backward_connections += 1

    cdef setLayer(self, Layer layer ):
        self.layer = layer

    cpdef setData(self, double x ):
        self.x = x

    cpdef setTarget(self, double t ):
        self.t = t

    cdef calculateZ(self):
        cdef:
            Connection connection
            list signals = []
        if not self.is_active_sum:
            self.is_active_sum = True
            self.z = 0.0
            if self.number_backward_connections > 0:
                self.z = 0.0
                for connection in self.backward_connections:
                    self.z += connection.activate()
                    signals.append( connection.activate() )
                #print "\nF => Neuron {}, Signal = {}\n".format( self.name, signals )
            else:
                self.z = self.x

    cpdef double activate(self):

        if not self.is_active_sum:
            self.calculateZ()
        if not self.is_active_state:
            self.is_active_state = True
            self.activateLayer()
            self.activationFunction()
        #print "\nF => Neuron {}, Y = {}, Z = {}\n".format( self.name, self.y, self.z )

        return self.y

    cdef activateLayer(self):
        pass

    cdef layerCalculations(self):
        pass

    cpdef calculated_dEdy(self):
        cdef:
            Connection connection
            list signals = []
        if not self.is_error_derivative_active:
            self.is_error_derivative_active = True
            signals = []
            self.dEdy = 0.0
            if self.number_forward_connections > 0:
                for connection in self.forward_connections:
                    self.dEdy += connection.errorActivate()
                signals = [ connection.errorActivate() for connection in self.forward_connections ]
                #print "\nB => Neuron {}, Signal = {}\n".format( self.name, signals )
    
    cdef dEdzFromNeurons(self):
        self.dEdz = self.dEdy * self.dydz()
    
    cdef calculate_dEdz(self):
        if self.number_forward_connections > 0:
            self.dEdzFromNeurons()
        else:
            self.dEdzFromTarget()

    cpdef double errorActivate(self):
        
        if not self.is_error_derivative_active:
            self.calculated_dEdy()
            
        if not self.is_local_gradient_active:
            self.is_local_gradient_active = True
            self.errorActivateLayer()
            self.calculate_dEdz()
            #print "\nB => Neuron {}, Local Gradient = {}, Function Derivative = {}, Error Derivative = {}\n".format( self.name, self.dEdz, self.dydz(), self.dEdy )

        return self.dEdz

    cdef errorActivateLayer(self):
        pass

    def errorLayerCalculations(self):
        pass

    cdef activationFunction(self):
        self.y = self.z

    cdef double dydz(self):
        return 1.0

    cdef double E(self):
        return 0.5*( self.t - self.y )**2

    cdef double dEdzFromTarget(self):
        self.dEdz = self.y - self.t

    cdef clear(self):
        self.is_active_sum = self.is_error_derivative_active = self.is_active_state = self.is_local_gradient_active = False
        self.y = self.z = self.dEdy = self.dEdz = 0.0

    cdef list getConnections(self):
        return self.forward_connections

    def __repr__(self):
        return str(self.y)

cdef class LogisticNeuron(Neuron):

    def __init__(self):
        Neuron.__init__(self)

    cpdef activationFunction(self):
        self.y = 1.0 / ( 1.0 + math.exp(-self.z ) )

    cpdef double dydz(self):
        return self.y - self.y**2

    cpdef double E(self):
        return -math.log(self.y) if self.t == 1 else -math.log( 1.0 - self.y )


cdef class TanhNeuron(Neuron):

    def __init__(self):
        Neuron.__init__(self)

    cpdef activationFunction(self):
        self.y = 2.0 / ( 1.0 + math.exp( -2.0*self.z ) ) - 1.0

    cpdef double dydz(self):
        return 1.0 - self.y**2

    cpdef double E(self):
        return -math.log( (self.y + 1.0) / 2.0 ) if self.t == 1 else -math.log( 1.0 - (self.y + 1.0) / 2.0 )

cdef class BiasUnit(Neuron):

    def __init__(self):
        Neuron.__init__(self)
        self.x = 1.0
        self.y = 1.0
        self.is_active_sum = True

    cdef activationFunction(self):
        self.y = 1.0

cdef class LayerActivatedNeuron(Neuron):

    def __init__(self, str name = ''):
        Neuron.__init__(self, name)

    cdef activateLayer(self):
        cdef:
            Neuron neuron

        if not self.layer.is_active:
            self.layer.is_active = True
            for neuron in self.layer.neurons:
                neuron.calculateZ()

            self.layerCalculations()

    cdef errorActivateLayer(self):
        cdef:
            Neuron neuron

        if not self.layer.is_error_active:
            self.layer.is_error_active = True
            for neuron in self.layer.neurons:
                neuron.calculated_dEdy()

            self.errorLayerCalculations()



cdef class SoftMaxNeuron(LayerActivatedNeuron):

    def __init__(self, str name = ''):
        Neuron.__init__(self, name)


    cdef layerCalculations(self):
        cdef:
            SoftMaxNeuron neuron

        self.layer.max = max( [ neuron.z for neuron in self.layer.neurons] )
        self.layer.sum = sum( [ math.exp( neuron.z - self.layer.max ) for neuron in self.layer.neurons] )


    cdef activationFunction(self):
        self.y = math.exp( self.z - self.layer.max ) / self.layer.sum
        
    cdef dEdzFromNeurons(self):
        cdef:
            SoftMaxNeuron neuron
            float dydz

        self.dEdz = 0.0
        for neuron in self.layer.neurons:
            dydz = self.y*(1.0 -self.y) if neuron is self else -self.y * neuron.y
            self.dEdz += neuron.dEdy * dydz



cdef class Connection(object):
    """
    Connection base class
    """
    cdef:
        public str name
        public Neuron source
        public Neuron destination
        double * _weight
        double * _weight_diff


    def __init__(self, Neuron source, Neuron destination, double weight = 0.0):

        self.name = source.name + "_" + destination.name

        self._weight = <double *> malloc( sizeof(double *) )
        self._weight_diff = <double *> malloc( sizeof(double *) )

        self.source = source
        source.addForwardConnection( self )

        self.destination = destination
        destination.addBackwardConnection( self )

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

        def __set__(self, double value):
            self._weight_diff[0] = value

    cdef double activate(self):
        cdef:
            double signal = self.source.activate()
            double product = signal * self.weight

        #print "    Connection {}, Weight {}, Signal {}, Product {}".format( self.name, self.weight, signal, product )
        return product

    cdef double errorActivate(self):
        cdef:
            double signal = self.destination.errorActivate()
            double product = signal * self.weight

        self.weight_diff = signal * self.source.y

        #print "    Connection {}, Product {}, Weight {}, Signal {}, Source State = {}, Weight Diff = {}".format( self.name, product, self.weight, signal, self.source.y, self.weight_diff )
        return signal * self.weight

    def __repr__(self):
        return str(self.weight)

cdef class Layer(object):
    cdef:
        public list neurons

    def __init__(self, *args, neuron_type = Neuron, list names = [] ):
        cdef:
            list type_list
            int i
            Neuron neuron
        self.neurons = list()
        if len(args) == 2:
            type_list = args[0]*[ args[1] ]
        elif type(args[0]) is int:
            type_list = args[0] * [neuron_type]
        else:
            type_list = args[0]
        for neuron_type in type_list:
            neuron = neuron_type()
            neuron.layer = self
            self.neurons.append( neuron )

        if names:
            for i in range(len(names)):
                self.neurons[i].name = names[i]

    def addNeuron(self, Neuron neuron ):
        self.neurons.append( neuron )
        neuron.setLayer(self)

    cpdef removeNeuron(self, Neuron neuron):
        self.neurons.remove(neuron)
        neuron.setLayer( None )

    cpdef list getConnections(self):
        cdef:
            list connections = list()
            Neuron neuron
        for neuron in self.neurons:
            connections += neuron.getConnections()
        return connections

    cpdef clear(self):
        cdef:
            Neuron neuron
        for neuron in self.neurons:
            neuron.clear()

    cdef double error(self):
        cdef:
            double error = 0.0
            Neuron neuron
        for neuron in self.neurons:
            error += neuron.E()

        return error

    #TODO: activate, errorActivate

    cpdef activate(self):
        cdef:
            Neuron neuron
        for neuron in self.neurons:
            neuron.activate()

    cpdef errorActivate(self):
        cdef:
            Neuron neuron
        for neuron in self.neurons:
            neuron.errorActivate()

    cpdef setData(self, double[:] data ):
        cdef:
            Neuron neuron
            int i = 0
        if len(self.neurons) != len(data):
            raise IndexError("data and layer dimensions are not equal")
        for neuron in self.neurons:
            neuron.setData( data[i] )
            i += 1

    cpdef setTarget(self, double[:] target ):
        cdef:
            Neuron neuron
            int i = 0

        if len(self.neurons) != len(target):
            raise IndexError("target and layer dimensions are not equal")
        for neuron in self.neurons:
            neuron.setTarget( target[i] )
            i += 1

    cpdef list connectTo( self, Layer b, connection = Connection ):

        cdef:
            list connection_list = list()
            list neuron_connexions
            Neuron na, nb

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

cdef class NeuronActivatedLayer(Layer):
    cdef:
        public bint is_active
        public bint is_error_active

    def __init__(self, *args, neuron_type = Neuron, list names = [] ):
        Layer.__init__(self,*args, neuron_type = neuron_type, names = names )
        self.is_active = self.is_error_active = False

    cpdef clear(self):
        Layer.clear(self)
        self.is_active = self.is_error_active = False


cdef class SoftMaxLayer(NeuronActivatedLayer):
    cdef:
        public float max
        public float sum

    def __init__(self, *args, neuron_type = SoftMaxNeuron, list names = [] ):
        NeuronActivatedLayer.__init__(self,*args, neuron_type = SoftMaxNeuron, names = names )
        self.max = -1000000000000.0
        self.sum = 0.0

    cpdef clear(self):
        NeuronActivatedLayer.clear(self)
        self.max = -1000000000000.0
        self.sum = 0.0


cdef class Network(object):

    cdef:
        public list input_layers
        public list output_layers
        public list auto_inputs
        public list fake_outputs
        public list layers

    def __init__(self):
        self.input_layers = list()
        self.output_layers = list()
        self.auto_inputs = list()
        self.fake_outputs = list()
        self.layers = list()

    cpdef addLayer(self, Layer layer):
        self.layers.append(layer)

    cpdef addInputLayer(self, Layer layer):
        self.input_layers.append(layer)

    cpdef addOutputLayer(self, Layer layer):
        self.output_layers.append(layer)

    cpdef addAutoInputLayer(self, Layer layer):
        self.auto_inputs.append(layer)

    cpdef addFakeOutputLayer(self, Layer layer):
        self.fake_outputs.append(layer)

    cpdef list getConnections(self):
        cdef:
            Layer layer
            list connections = list()
        for layer in self.input_layers + self.layers + self.auto_inputs:
            connections += layer.getConnections()
        return connections

    cpdef activateWith(self, double[:] input_data, bint return_value = False ):
        cdef:
            Layer layer
            Neuron neuron
            int neuron_count = 0
            int layer_length, i
            list values
            float[:] output

        self.clearLayers()
        for layer in self.input_layers:
            layer_length = len(layer.neurons)
            layer.setData(input_data[neuron_count:neuron_count+layer_length])
            neuron_count += layer_length

        if len(input_data) != neuron_count:
            raise IndexError("Input dimension dont match the number of input neurons")

        for layer in self.output_layers + self.fake_outputs:
            layer.activate()

        if return_value:
            return self.output_layers

    cpdef backpropagateWith(self, double[:] target_data ):
        cdef:
            Layer layer
            int neuron_count = 0
            int layer_length

        for layer in self.output_layers:
            layer_length = len(layer.neurons)
            layer.setTarget( target_data[neuron_count:neuron_count+layer_length] )
            neuron_count += layer_length

        if len(target_data) != neuron_count:
            raise IndexError("Target dimension dont match the number of output neurons")

        for layer in self.input_layers + self.auto_inputs:
            layer.errorActivate()

    cpdef clearLayers(self):
        cdef Layer layer
        for layer in self.output_layers + self.input_layers + self.layers + self.auto_inputs + self.fake_outputs:
            layer.clear()

    cdef double error(self, double[:] input_data, double[:] target_data ):
        cdef:
            double error = 0.0
            Neuron neuron
            Layer layer

        self.clearLayers()
        self.activateWith(input_data)
        self.backpropagateWith(target_data)
        for layer in self.output_layers + self.fake_outputs:
            error += layer.error()

        return error


    cpdef double[:] getGradient(self, double[:] input_data, double[:] target_data ):
        cdef:
            list connections = self.getConnections()
            Connection connection
            double[:] gradient
            int i

        gradient = array(shape=(len(connections),), itemsize=sizeof(double), format="d")
        self.clearLayers()
        self.activateWith(input_data)
        self.backpropagateWith(target_data)

        for i in range(len(connections)):
            gradient[i] = connections[i].dw

        return gradient


    cpdef double[:] numGradient(self, double[:] input_data, double[:] target_data, double delta = 0.01 ):
        cdef:
            list connections = self.getConnections()
            list gradient
            Connection connection
            double[:] num_gradient
            double e1, e2
            int i

        gradient = self.getConnections()
        num_gradient = array(shape=(len(gradient),), itemsize=sizeof(double), format="d")

        for i in range(len(gradient)):
            connections[i].w += delta
            e2 = self.error(input_data,target_data)

            connections[i].w -= 2.0*delta
            e1 = self.error(input_data,target_data)

            num_gradient[i] = ( e2 - e1 ) / ( 2.0*delta )

        return num_gradient



cdef class Trainer(object):

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
        cdef:
            Connection connection
            list connections
            int i = 0
        self.net = net
        self.input_data = input_data
        self.output_data = output_data
        self.learning_rate = learning_rate
        connections = net.getConnections()
        self.len_gradient = len(connections)
        self.len_data = len(input_data)
        self.total_gradient = array(shape=(self.len_gradient,), itemsize=sizeof(double), format="d")
        for connection in connections:
            self._weights.push_back(connection._weight)
            self._gradient.push_back(connection._weight_diff)

    cdef restartGradient(self):
        cdef:
            int i
        for i in range(self.len_gradient):
            self.total_gradient[i] = 0.0

    cdef printGradient(self):
        cdef:
            int i
        for i in range(self.len_gradient):
            #print(self.total_gradient[i],)
            pass

    cdef printActualGradient(self):
        cdef:
            int i
        for i in range(self.len_gradient):
            #print(self._gradient[i][0],)
            pass

    cdef printWeights(self):
        cdef:
            int i
        for i in range(self.len_gradient):
            #print self._weights[i][0],
            pass
        ##print

    cdef addToGradient( self ):
        cdef:
            int i
        for i in range(self.len_gradient):
            self.total_gradient[i] += self._gradient[i][0]

    cpdef epochs(self, int epochs):
        cdef:
            int i
        for i in range(epochs):
            self.fullBatch()

    cdef fullBatch(self):
        cdef:
            int i
        self.restartGradient()
        for i in range(self.len_data):
            self.net.activateWith(self.input_data[i] )
            self.net.backpropagateWith( self.output_data[i] )
            self.addToGradient()

        for i in range(self.len_gradient):
            self._weights[i][0] -= self.learning_rate * self.total_gradient[i]
