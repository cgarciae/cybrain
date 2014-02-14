__author__ = 'Cristian'
from cybrain import Neuron, LogisticNeuron, Connection, Layer, Network, BiasUnit, Trainer, SoftMaxLayer
import numpy as np
from time import time

#CREATE NETWORK
nnet = Network()

#TRUTH TABLE (DATA)
X =     [[0.0,0.0]];    Y = [[0.0, 1.0]]
X.append([1.0,0.0]); Y.append([1.0, 0.0])
X.append([0.0,1.0]); Y.append([1.0, 0.0])
X.append([1.0,1.0]); Y.append([0.0, 1.0])


#CONVERT DATA TO NUMPY ARRAY
X, Y = np.array(X), np.array(Y)

Lin = Layer( 2, names =['1','2'])
Lout = Layer( 2, LogisticNeuron, names= ['3', '4'] )
Lhidden = Layer( 2, LogisticNeuron, names= ['5','6'] )
bias = Layer( 1, BiasUnit, names=['b'] )

#ADD LAYERS TO NETWORK
nnet.addInputLayer(Lin)
nnet.addOutputLayer(Lhidden)
nnet.addLayer(Lout)
nnet.addAutoInputLayer(bias)

#CONNECT LAYERS
Lin.connectTo(Lout)
Lout.connectTo(Lhidden)
bias.connectTo(Lhidden)
bias.connectTo(Lout)

#CREATE BATCH TRAINER
rate = 0.1
batch = Trainer( nnet, X, Y, rate )

#TRAIN FOR 1000 EPOCHS
t1 = time()
batch.epochs(2000)
print "Time CyBrain {}".format(time()-t1)

#PRINT RESULTS
for x in X:
    print "{} ==> {}".format( x, nnet.activateWith(x, return_value= True) )
