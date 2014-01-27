import time as ti



from cybrain import LinearNeuron as N
from cybrain import *



t1 = ti.time()

Lin = Layer(3)
Lout = Layer(1)

neurons = Lin.neurons + Lout.neurons


fc = fullConnection(Lin,Lout)

connection_list = [ n.outgoing_connections[0] for n in Lin.neurons ]

lst = [Lin,Lout]
learning_rate = 0.1
target = 1.5
inpt = 1.0

x = LinearNeuron()
x.incoming_connection_count
x.forward_counter

for i in range(20):
    
    Lin.propagateInput([1.0,2.0,3.0])
    Lout.propagateErrorDerivative([target])
    
    #print( "a's activation state: {}, c's weight: {}, b's activation state: {}".format(Lin.neurons[0].activation_state, [ w.weight for w in connection_list ], Lout.neurons[0].activation_state) )

    for c in connection_list:
        c.weight -= learning_rate * c.weight_diff
        c.clearAcumulators()

    for x in neurons:
        x.clearAcumulators()
        x.clearCounters()
        #print(x.incoming_connection_count, x.forward_counter)


    print( Lout.neurons[0].outputError(target) )

print( "a's activation state: {}, c's weight: {}, b's activation state: {}".format(Lin.neurons[0].activation_state, [ w.weight for w in connection_list ], Lout.neurons[0].activation_state) )
print("Time: {:.3e}".format(ti.time() - t1) )

