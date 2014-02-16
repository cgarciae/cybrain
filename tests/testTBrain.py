__author__ = 'Cristian'
from tbrain import TNeuron, TLogisticNeuron, TConnection, TLayer, TNetwork, TBiasUnit, TTrainer
import numpy as np
from theano import shared, function, scan
from theano.tensor import dscalar, Sum, exp, log, grad, TensorConstant, mul, constant, stack
from theano.printing import pprint
nnet = TNetwork()

#TRUTH TABLE (DATA)
xin =     [[0.0,0.0]];    yout = [[0.0, 1.0]]
xin.append([1.0,0.0]); yout.append([1.0, 0.0])
xin.append([0.0,1.0]); yout.append([1.0, 0.0])
xin.append([1.0,1.0]); yout.append([0.0, 1.0])


#CONVERT DATA TO NUMPY ARRAY
xin, yout = np.array(xin), np.array(yout)

A = TLayer( 2, names =['1','2'])
B = TLayer( 2, TLogisticNeuron, names= ['3', '4'] )
C = TLayer( 2, TLogisticNeuron, names= ['5'] )
b = TLayer( 1, TBiasUnit, names=['b'] )

nnet.addInputLayer(A)
nnet.addOutputLayer(C)
nnet.addLayer(B)
nnet.addAutoInputLayer(b)

A_B = A.connectTo(B)
B_C = B.connectTo(C)
b.connectTo(B)
b.connectTo(C)

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


nnet.compile()


from time import time

t1 = time()
batch = TTrainer(nnet,xin,yout,0.1)
batch.epochs(2000)
for i in range(len(xin)):
        print "{} ==> {}".format( xin[i], nnet(*tuple(xin[i])) )

print "Time {}".format(time()-t1)

#nnet.compile()

# nnet.compile()

# for i in range(1000):
#     nnet.f_learn(1.0, 1.0, 0.0 )
#     print "States {}\n\nGradient {}\n\nWeights {}\n\n".format( nnet.f_states(1.0,1.0), nnet.f_gradient( 1.0, 1.0, 0.0 ), nnet.f_weights() )


# rate = 0.11

# batch = Trainer(nnet,xin,yout,rate)
# batch.epochs(1000)

# dw =  np.array( [c.weight_diff for c in connections] )*0.0
# for i in range(8000):
#     dw *= 0.0
#     for i in range(len(xin)):
#         nnet.activateWith( xin[i].copy() )
#         nnet.backpropagateWith( yout[i].copy() )
#         dw +=  np.array( [c.weight_diff for c in connections] )
#         # dw += np.array( nnet.numGradient(xin[j],yout[j], delta= 0.001) )
#         # dw += np.array( nnet.getGradient(xin[j],yout[j]) )
#         nnet.clearLayers()
#
#     # print "Gradient {}".format(np.array( nnet.getGradient(xin[1],yout[1]) ) )
#     # print "Num Grad {}\n".format(np.array( nnet.numGradient(xin[1],yout[1], delta= 0.01) ) )
#
#     for i in range(len(connections)):
#         connections[i].weight -= rate * dw[i]
#
#     L = []
#     for x in xin:
#         nnet.activateWith(x)
#         L.append(C.__repr__())
#     print "\n\n RESULTADOS \n\n"
#     print(L)

# L = []
# for x in xin:
#     nnet.activateWith(x)
#     print "{} ==> {}".format( x, nnet.output_layers )
#
#
# nnet.activateWith( xin[0].copy() )
# nnet.backpropagateWith( yout[0].copy() )

# w13, w14, w23, w24, w35, w45, w03, w04, w05 = connections[0].weight, connections[1].weight, connections[2].weight, connections[3].weight, connections[4].weight, connections[5].weight, connections[6].weight ,connections[7].weight, connections[8].weight
# w = [ w13, w14, w23, w24, w35, w45, w03, w04, w05 ]
#
# y0, y1, y2, t = 1.0, dscalar('y0'), dscalar('y1'), dscalar('t')
#
#
# z3 = y0*w03 + y1*w13 + y2*w23
#
# z4 = w04 + y1*w14 + y2*w24
# y3 = 1.0 / ( 1.0 + exp(-z3 ) )
# y4 = 1.0 / ( 1.0 + exp(-z4 ) )
#
# z5 = w05 + y3*w35 + y4*w45
# y5 = 1.0 / ( 1.0 + exp(-z5 ) )
#
# E = -t*log(y5) - (1.0 - t)*log(1.0-y5)
#
# dw = grad(E,w)
# dEdz = grad(E,[z3,z4,z5])
# dydz = [ grad( y3 , z3 ), grad(y4,z4)  ]
#
# fz = function( [y1,y2,t], [z3,z4,z5], on_unused_input='ignore' )
# fdw = function( [y1,y2,t], dw , on_unused_input='ignore')
# fy = function( [y1,y2,t], [y3,y4,y5] , on_unused_input='ignore')
# fdEdz = function( [y1,y2,t], dEdz , on_unused_input='ignore')
# fdydz = function( [y1,y2,t], dydz , on_unused_input='ignore' )


#
# vals = 1.0,1.0,0.0
#
#
# print "Weights = {}".format([c.weight for c in connections])
# print "States = {}".format( fy(*vals) )
# print "Z's: {}".format( fy(*vals) )
# print "Gradient: {}".format(fdw(*vals))
# print "Local Gradient: {}".format(fdEdz(*vals))
# #print "State Derivative: {}".format( fdydz(*vals))


