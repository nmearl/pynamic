# distutils: language=c++

from cpython cimport array
from libc.stdlib cimport free
from cpython cimport PyObject, Py_INCREF

# Import the Python-level symbols of numpy
import numpy as np

# Import the C-level symbols of numpy
cimport numpy as np

# Numpy must be initialized. When using numpy from C or Cython you must
# _always_ do that, or you will have segfaults
np.import_array()

# We need to build an array-wrapper class to deallocate our array when
# the Python object is deleted.

cdef class ArrayWrapper:
    cdef void* data_ptr
    cdef int size

    cdef set_data(self, int size, void* data_ptr):
        """ Set the data of the array
        This cannot be done in the constructor as it must recieve C-level
        arguments.
        Parameters:
        -----------
        size: int
            Length of the array.
        data_ptr: void*
            Pointer to the data            
        """
        self.data_ptr = data_ptr
        self.size = size

    def __array__(self):
        """ Here we use the __array__ method, that is called when numpy
            tries to get an array from the object."""
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.size
        # Create a 1D array, of length 'size'
        ndarray = np.PyArray_SimpleNewFromData(1, shape,
                                               np.NPY_INT, self.data_ptr)
        return ndarray

    def __dealloc__(self):
        """ Frees the array. This is called by Python when all the
        references to the object are gone. """
        free(<void*>self.data_ptr)


cdef extern from "n_body_state.h":
    cdef cppclass NBodyState:
        NBodyState(double* m, double posj[][3], double velj[][3], int NN, double t0)
        NBodyState(double* ms, double* a, double* e, double* inc, double* o, double* ln, double* m, int NN, double t0)
        double call "operator()"(double t, double H, double ORBIT_ERROR, double HLIMIT)
        double** getBaryLT()
        double V_Z_LT(int obj)
        # double occultn(double** pos, double* radii, double* u1, double* u2, double* fl, int N)
        
        
cdef class PyNBodyState:
    cdef NBodyState* ptr
    cdef int size

    def __cinit__(self, double ms, double a, double e, double inc, double o, double ln, double m, int NN, double t0):
        cdef array.array ms_arr = array.array('d', ms)
        cdef array.array a_arr = array.array('d', a)
        cdef array.array e_arr = array.array('d', e)
        cdef array.array inc_arr = array.array('d', inc)
        cdef array.array o_arr = array.array('d', o)
        cdef array.array ln_arr = array.array('d', ln)
        cdef array.array m_arr = array.array('d', m)

        size = len(ms_arr)

        self.ptr = new NBodyState(ms_arr.data.as_doubles, a_arr.data.as_doubles, e_arr.data.as_doubles, inc_arr.data.as_doubles, o_arr.data.as_doubles, ln_arr.data.as_doubles, m_arr.data.as_doubles, NN, t0)

    def __dealloc__(self):
        del self.ptr

    def __call__(self, double t, double h, double orbit_error, double h_limit):
        cdef double state = self.ptr.call(t, h, orbit_error, h_limit)

        return state

    def get_bary_lt(self):
        cdef double** lt = self.ptr.getBaryLT()
        cdef np.ndarray ndarray

        array_wrapper = ArrayWrapper()
        array_wrapper.set_data(self.size, <void*> lt)
        ndarray = np.array(array_wrapper, copy=False)
        # Assign our object to the 'base' of the ndarray object
        ndarray.base = <PyObject*> array_wrapper
        # Increment the reference count, as the above assignement was done in
        # C, and Python does not know that there is this additional reference
        Py_INCREF(array_wrapper)

        return ndarray

    def v_z_lt(self, int obj):
        cdef double lt = self.ptr.V_Z_LT(obj)

        return lt