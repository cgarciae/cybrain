__author__ = 'Cristian Garcia'

import cybrain2 as cb
import numpy as np
import unittest

class TestLayerFunctions(unittest.TestCase):

    def setUp (self):
        self.layerA = cb.Layer2(3)
        self.layerB = cb.Layer2(2)

        self.layerA.input = True

        self.layerA.setData(np.array([1., 2., 3.]))
        self.layerA.fullConnectionTo (self.layerB)

        for neuron in self.layerA.neurons:
            for (connection, value) in zip(neuron.forwardConnections, [1, 2]):
                connection.w = value


    def testConnectionNumber (self):
        listF = []
        for neuron in self.layerA.neurons:
            listF.extend (neuron.forwardConnections)

        listB = []
        for neuron in self.layerB.neurons:
            listB.extend (neuron.backwardConnections)

        self.assertEqual (len (listF), 6)
        self.assertEqual (len (listB), 6)


    def test_activationLayerB (self):
        self.layerB.activate()

        self.assertEqual(self.layerB.neurons[0].z, 6)
        self.assertEqual(self.layerB.neurons[0].y, 6)
        self.assertEqual(self.layerB.neurons[1].z, 12)
        self.assertEqual(self.layerB.neurons[1].y, 12)

        self.assertEqual(self.layerA.neurons[0].y, 1)
        self.assertEqual(self.layerA.neurons[1].y, 2)
        self.assertEqual(self.layerA.neurons[2].y, 3)


suite = unittest.TestLoader().loadTestsFromTestCase(TestLayerFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    unittest.main()
