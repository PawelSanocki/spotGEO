from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("myFilters.pyx"),
    include_dirs=[numpy.get_include()]
)
# cd myFilters
# python setup.py build_ext --inplace