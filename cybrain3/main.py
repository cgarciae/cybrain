__author__ = 'Cristian Garcia'

#import test
import numpy as np
import cybrain as cb


layerOut = cb.LinearLayer (3)
layerIn = cb.LinearLayer (2)
con = cb.FullConnection (2, 3)

layerIn._Y = np.array ([[1.,4.]])
con.W = np.array ([[3.,1.,1.],
                    [2.,1.,1.]])

layerOut.sourceConnections.append(con)
con.receiver = layerOut

layerIn.receiverConnections.append(con)
con.source = layerIn

layerIn.active = True

print np.array (layerOut.Y())