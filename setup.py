from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# os.environ["CC"] = "gcc-7"
# os.environ["CXX"] = "g++-7"

setup(
    ext_modules = cythonize(
        Extension(
            "phodyn",                  # our Cython source
            sources=["extern/phodyn.pyx", "extern/n_body_state.cpp", "extern/n_body.cpp"],  # additional source file(s)
            language="c++",                       # generate C++ code
            include_dirs=[np.get_include()]
        )
    ),
    name='pynamic',
    version='0.1.0',
    packages=['pynamic'],
    url='https://github.com/nmearl/pynamic',
    license='MIT',
    author='Nicholas Earl',
    author_email='contact@nicholasearl.me',
    description='N-body light curve modeling of multi-body systems.'
)
