import cybrain as cb
import numpy as np
from time import time

#TRUTH TABLE (DATA)
X =     [[0.0]];     Y = [[0.0]]
X.append([1.0]); Y.append([1.0])
X.append([2.0]); Y.append([1.0])
X.append([3.0]); Y.append([0.0])

#CONVERT DATA TO NUMPY ARRAY
X, Y = np.array(X), np.array(Y)

#CREATE NETWORK
nnet = cb.Network()

#CREATE LAYERS
Lin = cb.SparseLinearInt( 4, names= ['i1','i2','i3','i4'] )


Lhidden = cb.Layer( 2, cb.Neuron , names= ['c','d'] )
Lout = cb.Layer( 1, cb.LogisticNeuron , names= ['e'] )
bias = cb.Layer( 1, cb.BiasUnit, names= ['bias'] )

#ADD LAYERS TO NETWORK
nnet.addInputLayer(Lin)
nnet.addLayer(Lhidden)
nnet.addOutputLayer(Lout)
nnet.addAutoInputLayer(bias)
#
#CONNECT LAYERS
Lin.connectTo(Lhidden)
Lhidden.connectTo(Lout)
bias.connectTo(Lhidden)
bias.connectTo(Lout)


x = X[0]
nnet.activateWith(x)
print "x {}, Lout {}".format(x, [ (n.x,n.z,n.y) for n in Lout.neurons ] )

x = X[1]
nnet.activateWith(x)
print "x {}, Lout {}".format(x, [ (n.x,n.z,n.y) for n in Lout.neurons ] )

x = X[2]
nnet.activateWith(x)
print "x {}, Lout {}".format(x, [ (n.x,n.z,n.y) for n in Lout.neurons ] )

x = X[3]
nnet.activateWith(x)
print "x {}, Lout {}".format(x, [ (n.x,n.z,n.y) for n in Lout.neurons ] )

nnet.clearLayers()

#PRINT RESULTS
for x in X:
    nnet.activateWith(x)
    print "{} ==> {}, {}".format( x, Lout, [ (n.x,n.z,n.y) for n in Lout.neurons ] )

#CREATE BATCH TRAINER
rate = 0.1
batch = cb.Trainer( nnet, X, Y, rate )

#TRAIN
t1 = time()
batch.epochs(1000)
print "Time CyBrain {}".format(time()-t1)

#PRINT RESULTS
for x in X:
    print "{} ==> {}".format( x, nnet.activateWith(x, return_value= True) )