__author__ = 'Cristian'
#CYBRAIN
from cybrain import Neuron, LogisticNeuron, Connection, Layer, Network, BiasUnit, Trainer, SoftMaxLayer
import numpy as np
from time import time
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



print

#PYBRAIN
from pybrain.tools.shortcuts import buildNetwork
from pybrain import LinearLayer, SigmoidLayer, FeedForwardNetwork, FullConnection, BiasUnit, SoftmaxLayer
from pybrain.supervised.trainers.backprop import BackpropTrainer
from pybrain.structure.modules.tanhlayer import TanhLayer
from pybrain.datasets import SupervisedDataSet


ds = SupervisedDataSet(2,1 )

ds.addSample((0, 0), (0,))
ds.addSample((0, 1), (1,))
ds.addSample((1, 0), (1,))
ds.addSample((1, 1), (0,))


net = buildNetwork(2, 2, 1, bias=True, outputbias= True, hiddenclass=SigmoidLayer)
trainer = BackpropTrainer(net, ds, learningrate= 0.1)

t1 = time()
trainer.trainEpochs(2000)
print "Time PyBrain {}".format(time()-t1)

#PRINT RESULTS
for x in X:
    print "{} ==> {}".format( x, net.activate(x) )

