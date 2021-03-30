from distutils.core import setup, Extension
import numpy

#module = Extension('filters', include_dirs = ["C:\\Python\\Python37\\include"], libraries = [], sources = ["filtersmodule.cpp"])
module = Extension('filtersC', sources = ["filtersmodule.c"], include_dirs=[numpy.get_include()])
setup(name = "filtersC", version = "1.0", description = "", ext_modules = [module])