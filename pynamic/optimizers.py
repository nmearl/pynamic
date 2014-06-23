__author__ = 'nmearl'

import numpy as np
import emcee
import os
try:
    import pymultinest
except:
    pass
import scipy.optimize as op


def lnprior(dtheta, params):
    for i in range(len(dtheta)):
        if not (params[i].min <= dtheta[i] <= params[i].max):
            # print(params[i].name, params[i].min, dtheta[i], params[i].max)
            return -np.inf

    return 0.0


def lnlike(dtheta, optimizer, nprocs=1):
    optimizer.params.update(dtheta)
    mod_flux, mod_rv = optimizer.model(nprocs)

    f = optimizer.params.get("ferr_frac").value
    lnf = np.log(f) if f != 0.0 else -np.inf
    inv_sigma2 = 1.0 / (
    optimizer.photo_data[2] ** 2 + mod_flux ** 2 * np.exp(2 * lnf))
    flnl = -0.5 * (np.sum((optimizer.photo_data[1] - mod_flux) ** 2 * inv_sigma2
                          - np.log(inv_sigma2)))

    tinv_sigma2 = 1.0 / (
        optimizer.rv_data[2][:10] ** 2 + mod_rv[:10] ** 2 * np.exp(2 * -np.inf))
    trvlnl = -0.5 * (np.sum((optimizer.rv_data[1][:10] - mod_rv[:10]) ** 2 *
                            tinv_sigma2 - np.log(tinv_sigma2)))

    mrinv_sigma2 = 1.0 / (
        optimizer.rv_data[2][10:] ** 2 + mod_rv[10:] ** 2 * np.exp(2 * -np.inf))
    mrvlnl = -0.5 * (np.sum((optimizer.rv_data[1][10:] - mod_rv[10:]) ** 2 *
                            mrinv_sigma2 - np.log(mrinv_sigma2)))

    tlnl = np.sum(flnl) + np.sum(trvlnl) + np.sum(mrvlnl)

    return tlnl


def lnprob(dtheta, optimizer, nprocs=1):
    lp = lnprior(dtheta, optimizer.params.get_all(True))

    if not np.isfinite(lp):
        return -np.inf

    return lp + lnlike(dtheta, optimizer, nprocs)


def hammer(optimizer, nwalkers=None, niterations=500, nprocs=1):
    # Initialize the walkers
    if not nwalkers:
        nwalkers = len(optimizer.params.get_flat(can_vary=True)) ** 2

        if nwalkers % 2 != 0.0:
            nwalkers += 1

    theta = optimizer.params.get_flat(can_vary=True)
    ndim = len(theta)
    theta[theta == 0.0] = 1.0e-10
    pos0 = [theta + theta * 1.0e-3 * np.random.randn(ndim)
            for i in range(nwalkers)]

    # Setup the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(optimizer,),
                                    threads=nprocs)

    # Every iteration, save out chain
    for pos, lnp, state in sampler.sample(pos0, iterations=niterations,
                                          storechain=False):
        maxlnprob = np.argmax(lnp)
        bestpos = pos[maxlnprob, :]

        optimizer.iterout(lnp[maxlnprob], bestpos)

        for k in range(pos.shape[0]):
            if not np.isnan(lnp[k]):
                optimizer.update_chain(lnp[k], pos[k])


def minimizer(optimizer, method=None, nprocs=1):
    chi2 = lambda *args: -2 * lnprob(*args)
    theta = optimizer.params.get_flat(can_vary=True)
    result = op.minimize(chi2, theta, args=(optimizer, nprocs),
                         method=method)
    # bounds=optimizer.params.get_bounds(True))
    tlnl = -2 * lnlike(result["x"], optimizer)
    optimizer.iterout(tlnl, result["x"])
    optimizer.update_chain(tlnl, result["x"])


def multinest(optimizer, nprocs=1):
    # number of dimensions our problem has
    parameters = ["{0}".format(i)
                  for i in range(len(optimizer.params.get_all(True)))]
    nparams = len(parameters)

    if not os.path.exists('chains'):
        os.mkdir('chains')

    def lnprior(cube, ndim, nparams):
        theta = np.array([cube[i] for i in range(ndim)])

        for i in range(len(optimizer.params.get_all(True))):
            param = optimizer.params.get_all(True)[i]

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

        optimizer.params.update(theta)
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