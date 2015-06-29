from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
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



setup\
(
    cmdclass = {'build_ext': build_ext},

    ext_modules = [
        Extension( "cybrain",
        sources= ["cybrain.pyx"],
        include_dirs= [np.get_include()],
        language='c++'
        ),
        Extension( "test",
        sources= ["test.pyx"],
        include_dirs= [np.get_include()],
        language='c++'
        )]
)