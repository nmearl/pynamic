__author__ = 'nmearl'

import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer
import sys
from multiprocessing import Pool
import os

path = os.path.dirname(os.path.realpath(__file__)).split('/')[:-1]
path = '/'.join(map(str, path))

if sys.platform == 'darwin':
    # print("It seems you're on mac, loading mac libraries...")
    lib = ctypes.cdll.LoadLibrary(path + '/lib/photodynam-mac.so')
else:
    # print("It seems you're on linux, loading linux libraries...")
    lib = ctypes.cdll.LoadLibrary(path + '/lib/photodynam.so')

start = lib.start

start.argtypes = [
    ndpointer(ctypes.c_double),
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_double,
    ctypes.c_double,
    ctypes.c_double,
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ndpointer(ctypes.c_double),
    ctypes.c_int
]


def run(inputs):
    time, time_size, \
    N, t0, maxh, orbit_error, \
    masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma, \
    sub_flux, sub_rv, rv_body = inputs

    start(
        time, time_size,
        N, t0, maxh, orbit_error,
        masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma,
        sub_flux, sub_rv, rv_body
    )

    return sub_flux, sub_rv


def generate(params, time, rv_body, nprocs=1):
    N, t0, maxh, orbit_error = (int(params.get('nbodies').value),
                                params.get('epoch').value,
                                params.get('max_h').value,
                                params.get('orbit_error').value)

    masses = np.array([params.get('mass_{0}'.format(i)).value
                       for i in range(N)])
    radii = np.array([params.get('radius_{0}'.format(i)).value
                      for i in range(N)])
    fluxes = np.array([params.get('flux_{0}'.format(i)).value
                       for i in range(N)])
    u1 = np.array([params.get('u1_{0}'.format(i)).value for i in range(N)])
    u2 = np.array([params.get('u2_{0}'.format(i)).value for i in range(N)])
    a = np.array([params.get('a_{0}'.format(i)).value for i in range(1, N)])
    e = np.array([params.get('e_{0}'.format(i)).value for i in range(1, N)])
    inc = np.array([params.get('inc_{0}'.format(i)).value for i in range(1, N)])
    om = np.array([params.get('om_{0}'.format(i)).value for i in range(1, N)])
    ln = np.array([params.get('ln_{0}'.format(i)).value for i in range(1, N)])
    ma = np.array([params.get('ma_{0}'.format(i)).value for i in range(1, N)])

    time_chunks = np.array_split(time, nprocs)

    inputs = [
        [
            time_chunks[i], len(time_chunks[i]),
            N, t0, maxh, orbit_error,
            masses, radii, fluxes, u1, u2, a, e, inc, om, ln, ma,
            np.zeros(len(time_chunks[i])),
            np.zeros(len(time_chunks[i])),
            rv_body - 1
        ]
        for i in range(nprocs)
    ]

    if nprocs > 1:
        p = Pool(nprocs)

        result = p.map(run, inputs)

        p.close()
        p.join()
    else:
        result = map(run, inputs)

    result = np.array(result)

    return np.concatenate(result[:, 0]), np.concatenate(result[:, 1])