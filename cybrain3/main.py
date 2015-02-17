__author__ = 'Cristian Garcia'

import cybrain as cb

layerOut = cb.LinearLayer (3)
layerIn = cb.LinearLayer (2)
con = cb.FullConnection (2, 3)

layerOut.sourceConnections.append(con)
con.receiver = layerOut

layerIn.receiverConnections.append(con)
con.source = layerIn

layerIn.active = True



print "hola"
print layerOut.Y()