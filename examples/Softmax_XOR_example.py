import cybrain as cb
import numpy as np
from time import time

#TRUTH TABLE (DATA)
X =     [[0.0,0.0]];     Y = [[0.0,1.0]]
X.append([1.0,0.0]); Y.append([1.0,0.0])
X.append([0.0,1.0]); Y.append([1.0,0.0])
X.append([1.0,1.0]); Y.append([0.0,1.0])

#CONVERT DATA TO NUMPY ARRAY
X, Y = np.array(X), np.array(Y)

#CREATE NETWORK
nnet = cb.Network()

#CREATE LAYERS
Lin = cb.Layer( 2, names= ['a','b'] )
Lhidden = cb.Layer( 2, cb.LogisticNeuron , names= ['c','d'] )
Lout = cb.SoftMaxLayer( 2 , names= ['e', 'f'] )
bias = cb.Layer( 1, cb.BiasUnit, names= ['bias'] )

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

#CREATE BATCH TRAINER
rate = 0.1
batch = cb.Trainer( nnet, X, Y, rate )

#TRAIN
t1 = time()
batch.epochs(2000)
print "Time CyBrain {}".format(time()-t1)

#PRINT RESULTS
for x in X:
    print "{} ==> {}".format( x, nnet.activateWith(x, return_value= True) )