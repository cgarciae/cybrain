import time as ti

t1 = ti.time()

from cybrain import LinearNeuron as N
from cybrain import LinearConnection
from cybrain import Layer
a,b = N(is_input=True), N(is_output=True)


c = LinearConnection(a, b)

lst = [a,b,c]
learning_rate = 0.5
target = 1.5
inpt = 1.0

Lin = Layer([a])
Lout = Layer([b])

for i in range(10):
    
    Lin.forwpropagateInput([1.0])
    Lout.backpropagateError([1.5], learning_rate )
    
    print( "a's activation state: {}, c's weight: {}, b's activation state: {}".format(a.activation_state, c.weight, b.activation_state) )
    
    c.weight -= learning_rate * c.weight_diff
    
    c.clear()
    for x in [a,b]:
        x.clearAcumulators()
        x.clearCounters()

        
print("Time: {:.3e}".format(ti.time() - t1) )