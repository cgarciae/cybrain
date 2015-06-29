CyBrain
=======

Neural Networks in Cython, inspired by PyBrain, but focused on speed.

Check examples/logistic_XOR_example.py for a first approach.

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
        print "{} ==> {}".format(X[i], nnet.activate(X[i:i+1,:]))



Same Example in PyBrain
========================

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


About the Author
================

My name is Cristian Garcia, I am a Mathematical Engineer and Software developer, so I claim to be a Mathematical Developer.
I live in Colombia but plan to travel abroad as soon as I get my collage degree to do a Master in Artificial Intelligence
or get a job as a Mathematical Developer.
If you happen to need a Mathematical Developer at your company and find my code reasonable, contact me. 