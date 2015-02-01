__author__ = 'Cristian Garcia'

import cybrain as cb
import numpy as np
import unittest

class TestLayerFunctions(unittest.TestCase):

    def setUp (self):
        self.layerA = cb.InputLayer2(3)
        self.layerB = cb.Layer2(2)

        self.layerA.setData(np.array([1., 2., 3.]))

        for neuron in self.layerA.neurons:
            for (connection, value) in zip(neuron.forwardConnections, [1, 2]):
                connection.w = value

        self.layerA.fullConnectionTo (self.layerB)

        self.layerB.activate()

        print self.layerB.neurons[0].y


    def testConnectionNumber (self):
        listF = []
        for neuron in self.layerA.neurons:
            listF.extend (neuron.forwardConnections)

        listB = []
        for neuron in self.layerB.neurons:
            listB.extend (neuron.backwardConnections)

        self.assertEqual (len (listF), 6)
        self.assertEqual (len (listB), 6)
        print "aca tambien"

    def connection_test (self):
        print "aca"
        self.assertEqual(self.layerA, self.layerB.backwardLayers[0])


if __name__ == '__main__':
    unittest.main()
