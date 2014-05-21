from pynamic import photometry

__author__ = 'nmearl'

# Base imports
import numpy as np
import os
# External optimizer imports
import emcee
import pymultinest


class Optimizer(object):

    def __init__(self, params, photo_data_file='', rv_data_file='', rv_body=1, output_prefix=''):
        self.params = params
        self.photo_data = np.loadtxt(photo_data_file, unpack=True, usecols=(0, 1, 2))
        self.rv_data = np.loadtxt(rv_data_file, unpack=True, usecols=(0, 1, 2)) \
            if rv_data_file else np.zeros((3, 0))
        self.rv_body = rv_body
        self.output_prefix = output_prefix

        self.chain = np.empty([1, 1 + len(self.params.all(True))])

        self.maxlnp = 0.0
        self.bestpos = np.zeros(len(self.params.all(True)))
        self.redchisq = 0.0

    def run(self, method, **kwargs):
        if method == 'emcee':
            self.maxlnp = -np.inf
            self.hammer(**kwargs)
        elif method == 'multinest':
            self.maxlnp = -np.inf
            self.multinest(**kwargs)

    def save(self):
        np.save("chain_" + self.output_prefix, self.chain)

    def model(self, nprocs=1):
        flux_x, rv_x = self.photo_data[0], self.rv_data[0]
        x = np.append(flux_x, rv_x)
        x = np.unique(x[np.argsort(x)])

        flux_inds = np.in1d(x, flux_x, assume_unique=True)
        rv_inds = np.in1d(x, rv_x, assume_unique=True)

        mod_flux, mod_rv = photometry.generate(self.params, x, self.rv_body, nprocs)

        return mod_flux[flux_inds], mod_rv[rv_inds]

    def _iterout(self, tlnl, theta, mod_flux):
        self.maxlnp = tlnl
        self.bestpos = theta
        nbodies = int(self.params.get("nbodies").value)
        self.redchisq = np.sum(((self.photo_data[1] - mod_flux) / self.photo_data[2]) ** 2) /\
                        (self.photo_data[1].size - 1 - (nbodies * 5 + (nbodies - 1) * 6))

    def hammer(self, nwalkers=62, niterations=2, nprocs=1):

        def lnprior(dtheta, params):
            for i in range(len(dtheta)):
                all_params = params.all(True)

                if not (all_params[i].min <= dtheta[i] <= all_params[i].max):
                    return -np.inf

            return 0.0

        def lnlike(dtheta, params, nprocs):
            # if np.isnan(np.array(theta)).any():
            #     return -np.inf

            params.update_parameters(dtheta)
            mod_flux, mod_rv = self.model(nprocs)

            flnl = (-0.5 * ((mod_flux - self.photo_data[1]) / self.photo_data[2])**2)
            rvlnl = (-0.5 * ((mod_rv - self.rv_data[1]) / self.rv_data[2])**2)
            tlnl = np.sum(flnl) + np.sum(rvlnl)

            # iterprint(theta, np.sum(flnl) + np.sum(rvlnl), output_name)
            # Check to see if this is the best position
            if tlnl > self.maxlnp:
                self._iterout(tlnl, dtheta, mod_flux)

            return np.sum(flnl) + np.sum(rvlnl)

        def lnprob(dtheta, params, nprocs):
            lp = lnprior(dtheta, params)

            if not np.isfinite(lp):
                return -np.inf

            return lp + lnlike(dtheta, params, nprocs)

        # Initialize the walkers
        theta = self.params.get_flat(can_vary=True)
        ndim = len(theta)
        theta[theta == 0.0] = 1.0e-10
        pos0 = [theta + theta * 1.0e-3 * np.random.randn(ndim)
                for i in range(nwalkers)]

        # Setup the sampler
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,
                                        args=(self.params, nprocs))

        # Every iteration, save out chain
        for pos, lnp, state in sampler.sample(pos0, iterations=niterations, storechain=False):
            maxlnprob = np.argmax(lnp)
            bestpos = pos[maxlnprob, :]
            self.params.update_parameters(bestpos)

            for k in range(pos.shape[0]):
                if not np.isnan(lnp[k]):
                    nobj = np.append(lnp[k], pos[k])
                    self.chain = np.vstack([self.chain, nobj])

    def multinest(self, nprocs=1):
        # number of dimensions our problem has
        parameters = ["{0}".format(i) for i in range(len(self.params.all(True)))]
        nparams = len(parameters)

        if not os.path.exists('chains'):
            os.mkdir('chains')

        def lnprior(cube, ndim, nparams):
            theta = np.array([cube[i] for i in range(ndim)])

            for i in range(len(self.params.all(True))):
                param = self.params.all(True)[i]

                if "mass_" in param.name:
                    theta[i] = 10**(theta[i]*8 - 9)
                elif "radius_" in param.name:
                    theta[i] = 10**(theta[i]*4 - 4)
                elif "flux_" in param.name:
                    theta[i] = 10**(theta[i]*4 - 4)
                elif "a_" in param.name:
                    theta[i] = 10**(theta[i]*2 - 2)
                elif "e_" in param.name:
                    theta[i] = 10**(theta[i]*3 - 3)
                elif "inc_" in param.name:
                    theta[i] *= 2.0 * np.pi
                elif "om_" in param.name:
                    theta[i] = 2.0 * np.pi * 10**(theta[i]*2 - 2)
                elif "ln_" in param.name:
                    theta[i] = 2.0 * np.pi * 10**(theta[i]*8 - 8)
                elif "ma_" in param.name:
                    theta[i] = 2.0 * np.pi * 10**(theta[i]*2 - 2)

            for i in range(ndim):
                cube[i] = theta[i]

        def lnlike(cube, ndim, nparams):
            theta = np.array([cube[i] for i in range(ndim)])

            self.params.update_parameters(theta)
            mod_flux, mod_rv = self.model(nprocs)

            flnl = -(0.5 * ((mod_flux - self.photo_data[1]) / self.photo_data[2])**2)
            rvlnl = -(0.5 * ((mod_rv - self.rv_data[1]) / self.rv_data[2])**2)
            tlnl = np.sum(flnl) + np.sum(rvlnl)

            nobj = np.append(np.sum(flnl) + np.sum(rvlnl), theta)
            self.chain = np.vstack([self.chain, nobj])

            if tlnl > self.maxlnp:
                self._iterout(tlnl, theta, mod_flux)

            return np.sum(flnl) + np.sum(rvlnl)

        # run MultiNest
        pymultinest.run(lnlike, lnprior, nparams, n_live_points=1000)
