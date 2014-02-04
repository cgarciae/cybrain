CyBrain
=======

Neural Networks in Cython, inspired by PyBrain.

Check XOR_example.py for a first approach. Documentation not available yet!

XOR Example
===========

    import numpy as np
    from cybrain import Layer, LogisticNeuron, BiasUnit, fullConnection, Network, Trainer
    
    #TRUTH TABLE (DATA)
    xin =     [[0.0, 0.0]];    yout = [[0.0]]
    xin.append([1.0,0.0]); yout.append([1.0])
    xin.append([0.0,1.0]); yout.append([1.0])
    xin.append([1.0,1.0]); yout.append([0.0])
    
    #CONVERT DATA TO NUMPY ARRAY
    xin, yout = np.array(xin), np.array(yout)
    
    #CREATE NEURAL NETWORK
    nnet = Network()
    
    #CREATE LAYERS
    Lin = Layer(2)
    Lhidden = Layer( 4, LogisticNeuron )
    Lout = Layer( 1, LogisticNeuron)
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
    
    #CREATE FULLBATCH TRAINER
    learning_rate = 0.1
    batch = Trainer(nnet,xin,yout,learning_rate)
    
    #SHOW INPUT OUTPUT MAP, MAKE 1 IF GREATER THAN 0.5 ELSE 0
    print("Before Training")
    for i in xin:
        print "{} => {}".format(i,float(nnet.activate(i)[0] > 0.5))
        nnet.clearNetwork()
    
    #TRAINING
    batch.epochs(4000)
    
    #SHOW INPUT OUTPUT MAP, MAKE 1 IF GREATER THAN 0.5 ELSE 0
    print("\nAfter Training")
    for i in xin:
        print "{} => {}".format( i, float(nnet.activate(i)[0] > .5))
        nnet.clearNetwork()
    
    #PRINT
    print("\nCONNECTIONS:")
    print "BIAS TO HIDDEN {}".format(bias_hidden)
    print "INPUT TO HIDDEN {}".format(in_hidden)
    print "BIAS TO OUTPUT {}".format(bias_out)
    print "HIDDEN TO OUTPUT {}".format(hidden_out)

