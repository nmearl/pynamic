from distutils.core import setup, Extension

pdmodule = Extension(
    'photodynam',
    language = "c++",
    extra_compile_args=['-shared'],
    include_dirs = ['/usr/local/include'],
    library_dirs = ['/usr/local/lib'],
    sources = ['lib/photodynam.cpp', 'lib/n_body.cpp', 'lib/n_body_state.cpp',
               'lib/n_body_lc.cpp', 'lib/elliptic.c', 'lib/icirc.c',
               'lib/scpolyint.c', 'lib/mttr.cpp'])

setup(
    name='pynamic',
    version='0.1',
    packages=['pynamic'],
    url='http://github.com/nmearl/pynamic',
    license='MIT',
    author='Nicholas Earl',
    author_email='contact@nicholasearl.me',
    description='N-body simulation code for calculating photo-dynamics of systems.',
    ext_modules = [pdmodule],
    install_requires=[
        'scipy',
        'numpy',
        'matplotlib',
        'emcee'
    ]
)
