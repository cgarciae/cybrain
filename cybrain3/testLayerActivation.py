__author__ = 'Cristian Garcia'

import cybrain as cb
import numpy as np
import unittest


class TestLayerFunctions(unittest.TestCase):

    def setUp (self):

        self.layerOut = cb.LinearLayer (3)
        self.layerIn = cb.LinearLayer (2)
        self.con = cb.FullConnection (2, 3)

        self.layerOut.sourceConnections.append(self.con)
        self.con.receiver = self.layerOut

        self.layerIn.receiverConnections.append(self.con)
        self.con.source = self.layerIn

        self.layerIn.active = True

    def test_activation (self):
        self.layerIn._Y = np.array ([[1.,2.]])
        self.con.W = np.array ([[1.,2.,3.],
                                [4.,5.,6.]])

        result = np.array (self.layerOut.Y())

        self.assertEqual (result[0,0], 9)
        self.assertEqual (result[0,1], 12)
        self.assertEqual (result[0,2], 15)

    def test_activationLayerB (self):
        pass


suite = unittest.TestLoader().loadTestsFromTestCase(TestLayerFunctions)
unittest.TextTestRunner(verbosity=2).run(suite)

if __name__ == '__main__':
    unittest.main()
