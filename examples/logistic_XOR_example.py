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
Lhidden = cb.LinearLayer(2)
Lout = cb.LinearLayer(1)
bias = cb.LinearLayer(1)

#ADD LAYERS TO NETWORK
nnet.inputLayers.append(Lin)
nnet.outputLayers.append(Lout)
nnet.autoInputLayers.append(bias)

#CONNECT LAYERS
Lin.fullConnectTo(Lhidden)
Lhidden.fullConnectTo(Lout)
bias.fullConnectTo(Lhidden)
bias.fullConnectTo(Lout)

#SETUP NETWORK
nnet.setup()

#CREATE BATCH TRAINER
rate = 0.1
batch = cb.FullBatchTrainer(nnet, X, Y, rate)

#TRAIN
t1 = time()
batch.epochs(2000)
print "Time CyBrain {}".format(time()-t1)

#PRINT RESULTS
for i in range(len(X)):
    print "{} ==> {}".format(X[i], nnet.activate(X[i:i+1,:]))