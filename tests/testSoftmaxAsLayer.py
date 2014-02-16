__author__ = 'Cristian'
from theano import shared, function, scan
from theano.tensor import dscalar, Sum, exp, log, grad, TensorConstant, mul, constant, stack
import theano.tensor as T
from theano.printing import pprint
from cybrain import Neuron, LogisticNeuron, Connection, Layer, Network, BiasUnit, Trainer, SoftMaxLayer
import numpy as np
from time import time
nnet = Network()

#TRUTH TABLE (DATA)
X =     [[0.0,0.0]];     Y = [[0.0, 1.0]]
X.append([1.0,0.0]); Y.append([1.0, 0.0])
X.append([0.0,1.0]); Y.append([1.0, 0.0])
X.append([1.0,1.0]); Y.append([0.0, 1.0])


#CONVERT DATA TO NUMPY ARRAY
X, Y = np.array(X), np.array(Y)

#CREATE LAYERS
Lin = Layer( 2, names= ['a','b'] )
Lhidden = SoftMaxLayer( 2, names= ['c','d'] )
Lout = Layer( 2, LogisticNeuron , names= ['e', 'f'] )
bias = Layer( 1, BiasUnit, names= ['bias'] )

#ADD LAYERS TO NETWORK
nnet.addInputLayer(Lin)
nnet.addLayer(Lhidden)
nnet.addOutputLayer(Lout)
nnet.addAutoInputLayer(bias)

#CONNECT LAYERS
Lin.connectTo(Lhidden)
Lhidden.connectTo(Lout)
bias.connectTo(Lhidden)
bias.connectTo(Lout)

#Get Connections
connections = nnet.getConnections()
connections[0].weight = ( 0.3 )
connections[1].weight = ( -0.2 )
connections[2].weight = ( 0.11 )
connections[3].weight = ( -0.7 )
connections[4].weight = ( 0.435 )
connections[5].weight = ( -0.03 )
connections[6].weight = ( -0.5342 )
connections[7].weight = ( 0.83 )
connections[8].weight = ( -0.246 )
connections[9].weight = ( -0.5342 )
connections[10].weight = ( 0.83 )
connections[11].weight = ( -0.246 )

#CREATE BATCH TRAINER
nnet.activateWith(X[3])
nnet.backpropagateWith(Y[3])

# batch = Trainer(nnet,X,Y,0.1)
# batch.epochs(2000)
#
# #PRINT RESULTS
# for x in X:
#     print "{} ==> {}".format( x, nnet.activateWith(x, return_value= True) )


##Validation##

w13, w14, w23, w24, w35, w36, w45, w46, w03, w04, w05, w06 = shared(connections[0].weight), shared(connections[1].weight), shared(connections[2].weight), shared(connections[3].weight), shared(connections[4].weight), shared(connections[5].weight), shared(connections[6].weight) ,shared(connections[7].weight), shared(connections[8].weight), shared(connections[9].weight) ,shared(connections[10].weight), shared(connections[11].weight)
w = [ w13, w14, w23, w24, w35, w36, w45, w46, w03, w04, w05, w06 ]

y0, y1, y2, t1, t2 = 1.0, dscalar('y0'), dscalar('y1'), dscalar('t1'), dscalar('t2')


z3 = y0*w03 + y1*w13 + y2*w23
z4 = w04 + y1*w14 + y2*w24

S = T.exp(z3) + T.exp(z4)

y3 = T.exp(z3) / S
y4 = T.exp(z4) / S

z5 = w05 + y3*w35 + y4*w45
z6 = w06 + y3*w36 + y4*w46

y5 = 1.0 / ( 1.0 + exp(-z5 ) )
y6 = 1.0 / ( 1.0 + exp(-z6 ) )

E = - ( t1*T.log(y5) + (1.0-t1)*T.log(1.0-y5) + t2*T.log(y6) + (1.0-t2)*T.log(1.0-y6) )

dw = grad(E,w)
dEdz = grad(E,[z3,z4,z5,z6])
dydz = [ grad( y3 , z3 ), grad(y4,z4)  ]

fz = function( [y1,y2,t1,t2], [z3,z4,z5,z6], on_unused_input='ignore' )
fdw = function( [y1,y2,t1,t2], dw , on_unused_input='ignore')
fy = function( [y1,y2,t1,t2], [y3,y4,y5,y6] , on_unused_input='ignore')
fdEdz = function( [y1,y2,t1,t2], dEdz , on_unused_input='ignore')
fdydz = function( [y1,y2,t1,t2], dydz , on_unused_input='ignore' )



vals = 1.0,1.0,0.0,1.0


print "Weights = {}".format([c.weight for c in connections])
print "States = {}".format( fy(*vals) )
print "Z's: {}".format( fz(*vals) )
print "Gradient: {}".format(fdw(*vals))
print "Local Gradient: {}".format(fdEdz(*vals))
#print "State Derivative: {}".format( fdydz(*vals))

