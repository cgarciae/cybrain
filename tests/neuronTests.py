__author__ = 'Cristian Garcia'

import cybrain2 as cb
import numpy as np
import unittest

class TestNeuronFunctions(unittest.TestCase):

    def setUp (self):
        self.neuron = cb.Neuron2()

        self.neuron.active = True
        self.neuron._z = 2

    def testZproperty (self):
        self.assertEqual(self.neuron.z, self.neuron._z)

suite = unittest.TestLoader().loadTestsFromTestCase(TestNeuronFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    unittest.main()
