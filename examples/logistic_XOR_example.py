import sys
sys.path.append("..")

import cybrain as cb
import numpy as np
from time import time

#TRUTH TABLE (DATA)
X =     [[0.0,0.0]];     Y = [[0.0]]
X.append([1.0,0.0]); Y.append([1.0])
X.append([0.0,1.0]); Y.append([1.0])
X.append([1.0,1.0]); Y.append([0.0])

#CONVERT DATA TO NUMPY ARRAY
X, Y = np.array(X), np.array(Y)

#CREATE NETWORK
nnet = cb.Network()

#CREATE LAYERS
Lin = cb.LinearLayer(2)
Lhidden = cb.LogisticLayer(2)
Lout = cb.LogisticLayer(1)
bias = cb.BiasUnit()

#ADD LAYERS TO NETWORK
nnet.inputLayers = [Lin]
nnet.hiddenLayers = [Lhidden]
nnet.outputLayers = [Lout]
nnet.autoInputLayers = [bias]

#CONNECT LAYERS
Lin.fullConnectTo(Lhidden)
Lhidden.fullConnectTo(Lout)
bias.fullConnectTo(Lhidden)
bias.fullConnectTo(Lout)

#CREATE BATCH TRAINER
rate = 0.1
nnet.setup()
batch = cb.FullBatchTrainer(nnet, X, Y, rate)


#TRAIN
t1 = time()
batch.epochs(2000)
print "Time CyBrain {}".format(time()-t1)

#PRINT RESULTS
for i in range(len(X)):
    print "{} ==> {}".format(X[i], np.array(nnet.activate(X[i:i+1,:])))