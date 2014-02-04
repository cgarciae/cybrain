__author__ = 'Cristian'

import unittest
from math import exp

from cybrain import LogisticNeuron, LinearConnection, LinearNeuron


def similar(a,b,tol = 0.001):
    return abs(a-b) < tol

def sigmoid(z):
    return 1.0 / ( 1.0 + exp(-z) )

class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.Nin = LinearNeuron()
        self.Nin.weighted_sum = 2.0

        self.Nout = LinearNeuron()

        self.in_out = LinearConnection(self.Nin,self.Nout, 0.5)

        self.Logout = LogisticNeuron()

        self.in_Logout =  LinearConnection(self.Nin,self.Logout, 0.5)

    def test_linear_activation(self):
        self.Nin.propagateForward()
        a = self.Nout.state
        b = self.Nin.state * self.in_out.weight
        self.assertTrue( similar(a,b), "Failed " + self.test_linear_activation.__name__ )
        self.clear()

    def clear(self):
        self.Nin.clearAcumulators()
        self.Nin.clearCounters()
        self.Nout.clearAcumulators()
        self.Nout.clearCounters()

        self.in_out.clearAcumulators()
        self.in_Logout.clearAcumulators()

    def test_sigmoid_activation(self):
        self.Nin.propagateForward()
        a = self.Logout.state
        b = sigmoid( self.Nin.state * self.in_Logout.weight )
        self.assertTrue( similar(a,b), "Failed: a = {}, b = {}".format(a,b) )
        self.clear()

    def test_linear_backprop(self):
        target = 1.5
        self.Nin.propagateForward()
        self.Nout.errorDerivative(target)
        self.Nout.propagateErrorBackwards()

        a = self.in_out.weight_diff
        b = self.Nin.state * ( self.Nout.state - target )

        self.assertTrue( similar(a,b), "Failed: a = {}, b = {}".format(a,b) )
        self.clear()

    def test_linear_backprop(self):
        target = 1.5
        self.Nin.propagateForward()
        self.Nout.errorDerivative(target)
        self.Nout.propagateErrorBackwards()

        a = self.in_out.weight_diff
        b = self.Nin.state * ( self.Nout.state - target )

        self.assertTrue( similar(a,b), "Failed: a = {}, b = {}".format(a,b) )
        self.clear()

    def test_logistic_backprop(self):
        target = 0.0
        self.Nin.propagateForward()
        self.Logout.errorDerivative(target)
        self.Logout.propagateErrorBackwards()

        a = self.in_Logout.weight_diff
        b = self.Nin.state * ( self.Logout.state - target )

        self.assertTrue( similar(a,b), "Failed: a = {}, b = {}".format(a,b) )
        self.clear()

if __name__ == '__main__':
    unittest.main()
