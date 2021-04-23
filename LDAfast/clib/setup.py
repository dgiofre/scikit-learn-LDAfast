
import numpy 
numpy.get_include()

from Cython.Build import cythonize
from distutils.core import setup, Extension
# from Cython.Compiler.Options import directive_defaults

# directive_defaults['linetrace']=True
# directive_defaults['binding']=True

ext_modules = [
    Extension(
        "lda",
        ["lda.pyx"],
        # extra_compile_args=["-O3",'-ffast-math','-fopenmp'], #, '-pthread'
        # extra_link_args=['-lgomp'],  #'-fopenmp', 
    )  
]

setup( name = "Latent Dirichlet Allocation",
    ext_modules = cythonize(ext_modules),
    include_dirs=[numpy.get_include()]
)


