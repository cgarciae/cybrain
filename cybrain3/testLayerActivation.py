__author__ = 'Cristian Garcia'

import cybrain as cb
import numpy as np
import unittest


class TestLinearLayerFunctions(unittest.TestCase):

    def setUp (self):

        self.layerOut = cb.LinearLayer (3)
        self.layerIn = cb.LinearLayer (2)
        self.con = cb.FullConnection (self.layerIn, self.layerOut)

    def test_activation (self):
        self.layerIn.setData(np.array ([[1.,2.]]))
        self.con.setW (np.array ([[1.,2.,3.],
                                  [4.,5.,6.]]))

        result = self.layerOut.getY()

        np.testing.assert_almost_equal (result, [[9, 12, 15]])

    def test_dEdZ (self):
        self.layerIn.setData(np.array ([[1.,2.]]))
        self.con.setW (np.array ([[1.,2.,3.],
                                  [4.,5.,6.]]))

        result = self.layerOut.getY()
        self.layerOut.setTarget (np.array([[10., 14., 18.]]))
        self.layerIn.get_dEdZ()

        dEdZ = self.layerOut.get_dEdZ()
        dW = self.con.get_dW()

        np.testing.assert_almost_equal (dEdZ, [[-1, -2, -3]])
        np.testing.assert_almost_equal (dW, [[-1, -2, -3],
                                             [-2, -4, -6]])






class TestLogisticLayerFunctions(unittest.TestCase):

    def setUp (self):

        self.layerOut = cb.LogisticLayer (3)
        self.layerIn = cb.LinearLayer (3)
        self.connection = cb.LinearConnection (self.layerIn, self.layerOut)

    def test_activation (self):
        self.layerIn.setData(np.array ([[1.,-1., 0]]))
        self.connection.setW (np.array ([[1.,1.,1.]]))

        result = self.layerOut.getY()

        np.testing.assert_almost_equal (result, [[0.731059, 0.268941, 0.5]], decimal=5)

class TestLogisticLayerFunctions2(unittest.TestCase):

    def setUp (self):
        self.layerOut = cb.LogisticLayer (3)
        self.layerIn = cb.LinearLayer (2)
        self.connection = cb.FullConnection (self.layerIn, self.layerOut)

    def test_dEdZ (self):
        self.layerIn.setData(np.array ([[1.,2.]]))
        self.connection.setW (np.array ([[0.1,-0.2, 0.3],
                                         [-0.4, 0.5,-0.6]]))

        Y = self.layerOut.getY()
        self.layerOut.setTarget (np.array([[1., 0., 1.]]))
        self.layerIn.get_dEdZ()

        dW = self.connection.get_dW()

        np.testing.assert_almost_equal (Y, [[0.3318122278318339, 0.6899744811276125, 0.28905049737499605]])
        np.testing.assert_almost_equal (dW, [[-0.6681877721681662, 0.6899744811276126, -0.7109495026250039],
                                             [-1.3363755443363323, 1.3799489622552252, -1.4218990052500078]])


class TestLinearConnectionTests (unittest.TestCase):

    def setUp (self):
        self.net = cb.Network()

        self.l1 = cb.LinearLayer (3)
        self.l2 = cb.LogisticLayer (3)

        self.l3 = cb.LinearLayer (4)
        self.l4 = cb.LogisticLayer (3)
        self.l5 = cb.LinearLayer (6)

        self.l6 = cb.LogisticLayer (4)
        self.l7 = cb.LogisticLayer (8)

        self.net.inputLayers = [self.l1, self.l2]
        self.net.outputLayers = [self.l6, self.l7]

        self.l1.fullConnectTo(self.l3)
        self.l1.linearConnectTo(self.l4)
        self.l1.fullConnectTo(self.l5)

        self.l2.fullConnectTo(self.l3)
        self.l2.linearConnectTo(self.l4)
        self.l2.fullConnectTo(self.l5)

        self.l3.linearConnectTo(self.l6)
        self.l5.fullConnectTo(self.l7)

    def test_number_of_neurons (self):

        pass



class TestNetwork(unittest.TestCase):

    def setUp (self):
        self.net = cb.Network()

        self.l1 = cb.LinearLayer (3)
        self.l2 = cb.LogisticLayer (3)

        self.l3 = cb.LinearLayer (4)
        self.l4 = cb.LogisticLayer (3)
        self.l5 = cb.LinearLayer (6)

        self.l6 = cb.LogisticLayer (4)
        self.l7 = cb.LogisticLayer (8)

        self.net.inputLayers = [self.l1, self.l2]
        self.net.outputLayers = [self.l6, self.l7]

        self.l1.fullConnectTo(self.l3)
        self.l1.linearConnectTo(self.l4)
        self.l1.fullConnectTo(self.l5)

        self.l2.fullConnectTo(self.l3)
        self.l2.linearConnectTo(self.l4)
        self.l2.fullConnectTo(self.l5)

        self.l3.linearConnectTo(self.l6)
        self.l5.fullConnectTo(self.l7)

    def test_lenght (self):
        pass

    def test_findHiddenComponents (self):

        self.net.findHiddenComponents()

        self.assertEqual (len (self.net.layers), 7)
        self.assertEqual (len (self.net.connections), 8)

    def test_activate (self):
        self.net.activate(np.array([[1.,2.,3.,4.,5.,6.]]))





if __name__ == '__main__':
    space = "************************************************************************************************"


    print space
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLinearLayerFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)
    print space
    print "\n\n\n"

    print space
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLogisticLayerFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)
    print space
    print "\n\n\n"

    print space
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLogisticLayerFunctions2)
    unittest.TextTestRunner(verbosity=2).run(suite)
    print space
    print "\n\n\n"

    print space
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNetwork)
    unittest.TextTestRunner(verbosity=2).run(suite)
    print space
    print "\n\n\n"

    print space
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLinearConnectionTests)
    unittest.TextTestRunner(verbosity=2).run(suite)
    print space
    print "\n\n\n"
