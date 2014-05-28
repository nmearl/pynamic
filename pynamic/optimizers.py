__author__ = 'nmearl'

import numpy as np
import emcee
import os

try:
    import pymultinest
except:
    print("Unable to import PyMultinest")
import scipy.optimize as op


def lnprior(dtheta, params):
    for i in range(len(dtheta)):
        all_params = params.all(True)

        if not (all_params[i].min <= dtheta[i] <= all_params[i].max):
            return -np.inf

    return 0.0


def lnlike(dtheta, optimizer, nprocs=1):
    optimizer.params.update_parameters(dtheta)
    mod_flux, mod_rv = optimizer.model(nprocs)

    flnl = (-0.5 * ((mod_flux - optimizer.photo_data[1]) /
                    optimizer.photo_data[2]) ** 2)
    rvlnl = (-0.5 * ((mod_rv - optimizer.rv_data[1]) /
                     optimizer.rv_data[2]) ** 2)
    tlnl = np.sum(flnl) + np.sum(rvlnl)

    # Check to see if this is the best position
    if tlnl < abs(optimizer.maxlnp):
        optimizer.iterout(tlnl, dtheta, mod_flux)

    return tlnl


def lnprob(dtheta, optimizer):
    lp = lnprior(dtheta, optimizer.params)

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(dtheta, optimizer)


def hammer(optimizer, nwalkers=62, niterations=2, nprocs=1):
    # Initialize the walkers
    theta = optimizer.params.get_flat(can_vary=True)
    ndim = len(theta)
    theta[theta == 0.0] = 1.0e-10
    pos0 = [theta + theta * 1.0e-3 * np.random.randn(ndim)
            for i in range(nwalkers)]

    # Setup the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                    args=(optimizer.params, optimizer),
                                    threads=nprocs)

    # Every iteration, save out chain
    for pos, lnp, state in sampler.sample(pos0, iterations=niterations,
                                          storechain=False):
        maxlnprob = np.argmax(lnp)
        bestpos = pos[maxlnprob, :]
        optimizer.params.update_parameters(bestpos)

        for k in range(pos.shape[0]):
            if not np.isnan(lnp[k]):
                nobj = np.append(lnp[k], pos[k])
                optimizer.chain = np.vstack([optimizer.chain, nobj])


def minimizer(optimizer, method='SLSQP', nprocs=1):
    chi2 = lambda *args: -2 * lnlike(*args)
    theta = optimizer.params.get_flat(can_vary=True)
    result = op.minimize(chi2, theta, args=(optimizer, nprocs), method=method,
                         bounds=optimizer.params.get_bounds(True))
    optimizer.params.update_parameters(result["x"])


def multinest(optimizer, nprocs=1):
    # number of dimensions our problem has
    parameters = ["{0}".format(i)
                  for i in range(len(optimizer.params.all(True)))]
    nparams = len(parameters)

    if not os.path.exists('chains'):
        os.mkdir('chains')

    def lnprior(cube, ndim, nparams):
        theta = np.array([cube[i] for i in range(ndim)])

        for i in range(len(optimizer.params.all(True))):
            param = optimizer.params.all(True)[i]

            if "mass_" in param.name:
                theta[i] = 10 ** (theta[i] * 8 - 9)
            elif "radius_" in param.name:
                theta[i] = 10 ** (theta[i] * 4 - 4)
            elif "flux_" in param.name:
                theta[i] = 10 ** (theta[i] * 4 - 4)
            elif "a_" in param.name:
                theta[i] = 10 ** (theta[i] * 2 - 2)
            elif "e_" in param.name:
                theta[i] = 10 ** (theta[i] * 3 - 3)
            elif "inc_" in param.name:
                theta[i] *= 2.0 * np.pi
            elif "om_" in param.name:
                theta[i] = 2.0 * np.pi * 10 ** (theta[i] * 2 - 2)
            elif "ln_" in param.name:
                theta[i] = 2.0 * np.pi * 10 ** (theta[i] * 8 - 8)
            elif "ma_" in param.name:
                theta[i] = 2.0 * np.pi * 10 ** (theta[i] * 2 - 2)

        for i in range(ndim):
            cube[i] = theta[i]

    def lnlike(cube, ndim, nparams):
        theta = np.array([cube[i] for i in range(ndim)])

        optimizer.params.update_parameters(theta)
        mod_flux, mod_rv = optimizer.model(nprocs)

        flnl = -(0.5 * ((mod_flux - optimizer.photo_data[1]) /
                        optimizer.photo_data[2]) ** 2)
        rvlnl = -(0.5 * ((mod_rv - optimizer.rv_data[1]) /
                         optimizer.rv_data[2]) ** 2)
        tlnl = np.sum(flnl) + np.sum(rvlnl)

        nobj = np.append(np.sum(flnl) + np.sum(rvlnl), theta)
        optimizer.chain = np.vstack([optimizer.chain, nobj])

        if tlnl > optimizer.maxlnp:
            optimizer.iterout(tlnl, theta, mod_flux)

        return np.sum(flnl) + np.sum(rvlnl)

    # run MultiNest
    pymultinest.run(lnlike, lnprior, nparams, n_live_points=1000)