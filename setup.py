from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

# connections = Extension( "connections",
#                  sources= ["connections.pyx"],
#                  language='c++'
#                  )
# #  
# neurons = Extension( "neurons",
#                  sources= ["neurons.pyx"],
#                  language='c++'
#                  )
#  
# layers = Extension( "layers",
#                  sources= ["layers.pyx"],
#                  language='c++'
#                  )

cybrain = Extension( "cybrain",
                 sources= ["cybrain.pyx"],
                 include_dirs= [np.get_include()],
                 language='c++'
                 )
tbrain = Extension( "tbrain",
                 sources= ["tbrain.pyx"],
                 include_dirs= [np.get_include()],
                 language='c++'
                 )


setup(
    cmdclass = {'build_ext': build_ext},
#     ext_modules = [ neurons, connections, layers ]
    ext_modules = [ cybrain, tbrain ]
)