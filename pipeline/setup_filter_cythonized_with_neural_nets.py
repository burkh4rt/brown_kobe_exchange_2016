# CC=gcc-6 python setup_filter_cythonized.py build_ext --inplace
# python3 -c "import filter_cythonized_with_neural_nets"

# Cython has its own "extension builder" module that knows how
# to build cython files into python modules.
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext = Extension("filter_cythonized_with_neural_nets", sources=["filter_cythonized_with_neural_nets.pyx"],
        include_dirs = [numpy.get_include()],
        extra_compile_args = ["-fopenmp", "-g", "-O3"],
        extra_link_args=["-fopenmp", "-g"] )

setup(ext_modules=[ext],
    cmdclass={'build_ext': build_ext})
