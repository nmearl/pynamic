__author__ = 'nmearl'

from pynamic import photometry, optimizers
import numpy as np


class Optimizer(object):
    def __init__(self, params, photo_data_file='', rv_data_file='', rv_body=0,
                 chain_file=None):
        self.params = params
        self.photo_data = np.loadtxt(photo_data_file,
                                     unpack=True, usecols=(0, 1, 2))
        self.rv_data = np.zeros((3, 0))

        if rv_data_file:
            self.rv_data = np.loadtxt(rv_data_file,
                                      unpack=True, usecols=(0, 1, 2))

        self.rv_body = rv_body
        self.chain = np.zeros([1, 1 + len(self.params.all(True))])
        self.maxlnp = np.inf
        self.bestpos = np.zeros(len(self.params.all(True)))
        self.redchisq = 0.0

        if chain_file:
            self.chain = np.load(chain_file)

    def run(self, method=None, **kwargs):
        if method == 'mcmc':
            optimizers.hammer(self, **kwargs)
        elif method == 'multinest':
            optimizers.multinest(self, **kwargs)
        else:
            optimizers.minimizer(self, method=method, **kwargs)

    def save(self, chain_out_file="chain.txt", photo_out_file="photo_model.txt",
             rv_out_file="rv_model.txt"):
        if False:
            mod_flux, mod_rv = self.model()
            mod_rv = self.filled_rv_model()
            np.savetxt(photo_out_file, mod_flux)
            np.savetxt(rv_out_file, mod_rv)
            np.savetxt(chain_out_file, self.chain)

        np.save("chain", self.chain)

    def model(self, nprocs=1):
        flux_x, rv_x = self.photo_data[0], self.rv_data[0]
        x = np.append(flux_x, rv_x)
        x = np.unique(x[np.argsort(x)])

        flux_inds = np.in1d(x, flux_x, assume_unique=True)
        rv_inds = np.in1d(x, rv_x, assume_unique=True)

        mod_flux, mod_rv = photometry.generate(self.params, x,
                                               self.rv_body, nprocs)

        return mod_flux[flux_inds], mod_rv[rv_inds]

    def filled_rv_model(self, input_times, nprocs=1):
        mod_flux, mod_rv = photometry.generate(self.params, input_times,
                                               self.rv_body, nprocs)

        return mod_rv

    def iterout(self, tlnl=-np.inf, theta=None):
        if theta:
            self.params.save()
            self.save()
            self.bestpos = theta
            self.params.update_parameters(theta)

        mod_flux, mod_rv = self.model()
        chisq = np.sum(((self.photo_data[1] - mod_flux) /
                        self.photo_data[2]) ** 2)

        chisq += np.sum(((self.rv_data[1] - mod_rv) /
                         self.rv_data[2]) ** 2)

        deg = len(self.params.get_flat(True))
        nu = self.photo_data[1].size + self.rv_data[1].size - 1.0 - deg
        self.redchisq = chisq / nu

        # Update the maxlnp so the Watcher knows to print to screen
        self.maxlnp = tlnl