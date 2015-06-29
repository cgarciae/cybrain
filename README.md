CyBrain
=======

Neural Networks in Cython, inspired by PyBrain, but focused on speed.

Check Comparison.py for a first approach. In this example we train a CyBrain network to solve the XOR problem.

XOR Example
===========

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



Same Example in PyBrain
========================

    from pybrain.tools.shortcuts import buildNetwork
    from pybrain import LinearLayer, SigmoidLayer, FeedForwardNetwork, FullConnection, BiasUnit, SoftmaxLayer
    from pybrain.supervised.trainers.backprop import BackpropTrainer
    from pybrain.structure.modules.tanhlayer import TanhLayer
    from pybrain.datasets import SupervisedDataSet
    from time import time
    
    
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


Outputs
=======

    Time CyBrain 0.211654996872
    [ 0.  0.] ==> [ '0.0365560102866' ]
    [ 1.  0.] ==> [ '0.951081842587'  ]
    [ 0.  1.] ==> [ '0.951928021684'  ]
    [ 1.  1.] ==> [ '0.0332036251855' ]
    
    Time PyBrain 7.03572702408
    [ 0.  0.] ==> [  1.67662906e-08]
    [ 1.  0.] ==> [ 0.99999998]
    [ 0.  1.] ==> [ 0.99999998]
    [ 1.  1.] ==> [  7.30255101e-09]

Roadmap
========

* Implement other common Layers: Tanh, Softmax, etc [High].
* Refactor variable and methods according to python style guide [Medium].
* Cython memoryview operations optimization [High]
* Auto-encoders and CNNs [Requires research]
* Try to organize the folders [Cython research]
* Solutions to Andrew Ng's ML class with CyBrain [Soon]