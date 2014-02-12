import time as ti
import numpy as np
from cybrain import Layer, LogisticNeuron, BiasUnit, fullConnection, Network, Trainer, SoftMaxLayer

t1 = ti.time()

#TRUTH TABLE (DATA)
xin =     [[0.0, 0.0]];    yout = [[0.0, 1.0]]
xin.append([1.0,0.0]); yout.append([1.0, 0.0])
xin.append([0.0,1.0]); yout.append([1.0, 0.0])
xin.append([1.0,1.0]); yout.append([0.0, 1.0])


#CONVERT DATA TO NUMPY ARRAY
xin, yout = np.array(xin), np.array(yout)
print xin
print yout

#CREATE NEURAL NETWORK
nnet = Network()

#CREATE LAYERS
Lin = Layer(2)
Lhidden = Layer( 2, LogisticNeuron )
Lout = Layer( 1, LogisticNeuron )
Lbias = Layer( 1, BiasUnit )

#CREATE CONNECTIONS
in_hidden = fullConnection(Lin,Lhidden)
hidden_out = fullConnection(Lhidden,Lout)
bias_hidden = fullConnection(Lbias,Lhidden)
bias_out = fullConnection(Lbias,Lout)

#ADD LAYERS TO NETWORK
nnet.addInputLayer(Lin)
nnet.addOutputLayer(Lout)
nnet.addLayer(Lhidden)
nnet.addAutoInputLayer(Lbias)

#ADD CONNECTIONS TO NETWORK
nnet.addConnections(in_hidden)
nnet.addConnections(hidden_out)
nnet.addConnections(bias_hidden)
nnet.addConnections(bias_out)

connections = nnet.connections
connections[0].weight = 0.3
connections[1].weight = -0.2
connections[2].weight = 0.11
connections[3].weight = -0.7
connections[4].weight = 0.435
connections[5].weight = -0.03
connections[6].weight = -0.5342
connections[7].weight = 0.83
connections[8].weight = -0.246

nnet.activate(np.array([0.5,-0.5]))
nnet.backpropagateError(np.array([0.0]))



print "States: {}".format(nnet.neurons)
print "Weights: {}".format( [c.weight for c in connections ] )
print "Gradient: {}".format( [c.weight_diff for c in connections ] )
#print "Local Gradient: {}".format([ n.local_gradient for n in nnet.neurons ])
#print "State Derivative: {}".format([ n.dydz() for n in B.neurons + C.neurons ])


delta = 0.001
grad = []

for c in connections:
    nnet.clearNetwork()

    c.weight += delta
    nnet.activate(np.array([0.5,-0.5]))
    #nnet.backpropagateError(np.array([0.0]))
    v2 = Lout.neurons[0].outputError(0.0)

    nnet.clearNetwork()

    c.weight -= 2.0*delta
    nnet.activate(np.array([0.5,-0.5]))
    #nnet.backpropagateError(np.array([0.0]))
    v1 = Lout.neurons[0].outputError(0.0)

    c.weight += delta

    grad.append((v2-v1)/(2.0*delta))

print "Num Gradient: {}".format( grad )



# #CREATE FULLBATCH TRAINER
# learning_rate = 0.1
# batch = Trainer(nnet,xin,yout,learning_rate)
# print np.array(batch.output_data)
#
# #SHOW INPUT OUTPUT MAP, MAKE 1 IF GREATER THAN 0.5 ELSE 0
# print("Before Training")
# for i in xin:
#     print "{} => {}".format( i, np.array(nnet.activate(i)))
#     nnet.clearNetwork()
#
# #TRAINING
# batch.epochs(16000)
#
# #SHOW INPUT OUTPUT MAP, MAKE 1 IF GREATER THAN 0.5 ELSE 0
# print("\nAfter Training")
# for i in xin:
#     print "{} => {}".format( i, np.array(nnet.activate(i)))
#     nnet.clearNetwork()

#PRINT
# print("\nCONNECTIONS:")
# print "BIAS TO HIDDEN \n{}".format(np.array(bias_hidden))
# print "INPUT TO HIDDEN \n{}".format(np.array(in_hidden))
# print "BIAS TO OUTPUT \n{}".format(np.array(bias_out))
# print "HIDDEN TO OUTPUT \n{}".format(np.array(hidden_out))

#
# print "\nTime {}".format( ti.time() - t1 )

