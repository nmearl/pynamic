__author__ = 'nmearl'

import os
import numpy as np
import pylab


class Analyzer(object):

    def __init__(self, optimizer=None, path=None):
        self.optimizier = optimizer

        # burnin = int(0.5 * len(self.optimizier.chain))
        self.samples = self.optimizier.chain

        if path:
            self.samples = np.load(path)

        self.results = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                           zip(*np.percentile(self.samples[:, 1:], [16, 50, 84],
                                              axis=0)))

    def report(self):
        print("{0:12s} {1:12s} {2:12s} {3:12s}".format('Name', 'Value', 'Err-', 'Err+'))

        j = 0
        with open("report_" + self.optimizier.output_prefix + ".out", 'w') as f:
            for i in range(len(self.optimizier.params.all())):
                param = self.optimizier.params.all()[i]

                if param.vary:
                    print("{0:12s} {1:12g} {2:12g} {3:12g}".format(param.name, self.results[j][0],
                                                                   self.results[j][1], self.results[j][2]))

                    f.write("{0:12s} {1:12g} {2:12g} {3:12g}\n".format(param.name, self.results[j][0],
                                                                       self.results[j][1], self.results[j][2]))
                    j += 1
                else:
                    print("{0:12s} {1:12g}".format(param.name, param.value))

                    f.write("{0:12s} {1:12g}\n".format(param.name, param.value))

    def show(self, method, param_list=None):
        if method == 'histogram':
            self.histogram(param_list=param_list, show=True)
        elif method == 'flux':
            self.plot_flux()
        elif method == 'rv':
            self.plot_rv()
        elif method == 'chi':
            self.chi(param_list=param_list, show=True)

    def save(self, method, param_list=None, prefix='hist'):
        if os.path.exists("./plots"):
            os.mkdir("./plots")

        if method == 'histogram':
            self.histogram(save=True, show=False, param_list=param_list, prefix=prefix)
        elif method == 'flux':
            self.plot_flux(save=True, show=False)
        elif method == 'rv':
            self.plot_rv(save=True, show=False)
        elif method == 'chi':
            self.chi(param_list=param_list, save=True, show=False)

    def chi(self, param_list=None, show=True, save=False):

        for i in range(1, len(self.optimizier.params.all(True))):
            pylab.plot(self.samples[:, i], self.samples[:, 0], '+')
            pylab.show()

    def histogram(self, param_list=None, save=False, show=True, prefix='hist'):

        for i in range(1, len(self.optimizier.params.all(True))):
            param = self.optimizier.params.all(True)[i]

            if param_list and param.name not in param_list:
                continue

            ax = pylab.subplot('111')#.format(res, j+1))
            ax.hist(self.samples[:, i], 100, color="k", alpha=0.5, histtype="step")
            ax.axvline(self.results[i][0], color="r", lw=2)
            ax.axvline(self.results[i][0] + self.results[i][1], color='k', alpha=0.6)
            ax.axvline(self.results[i][0] - self.results[i][2], color='k', alpha=0.6)
            ax.set_title("{0:s} Parameter Distribution".format(param.name))
            ax.set_xlim(self.results[i][0] * 0.95, self.results[i][0] * 1.05)
            pylab.tight_layout()

            if save:
                pylab.savefig('./plots/{0}_{1:s}.png'.format(prefix, param.name))
            if show:
                pylab.show()

            pylab.close()

    def step(self, param_list=None, save=False, show=True, prefix='step'):

        for i in range(1, len(self.optimizier.params.all(True))):
            pylab.figure()

            param = self.optimizier.params.all(True)[i]

            if param_list and param.name not in param_list:
                continue

            ax = pylab.subplot('111')
            ax.plot(range(len(self.samples)), self.samples[:, i], color="k", alpha=0.5)
            ax.axhline(self.results[i][0], color="r", lw=2)
            ax.axhline(self.results[i][0] + self.results[i][1], color='k', alpha=0.6)
            ax.axhline(self.results[i][0] - self.results[i][2], color='k', alpha=0.6)
            ax.set_title("{0:s} Step".format(param.name))
            # ax.set_xlim(results[k][0] * 0.95, results[k][0] * 1.05)
            pylab.tight_layout()

            if save:
                pylab.savefig('./plots/{0}_{1:s}.png'.format(prefix, param.name))
            if show:
                pylab.show()

            pylab.close()

    def plot_flux(self, save=False, show=True, prefix='plot_flux'):
        mod_flux, mod_rv = self.optimizier.model()

        pylab.plot(self.optimizier.photo_data[0], self.optimizier.photo_data[1], 'k.')
        pylab.plot(self.optimizier.photo_data[0], mod_flux, 'r')

        if save:
            pylab.savefig('./plots/{0}.png'.format(prefix))
        if show:
            pylab.show()

        pylab.close()

    def plot_rv(self, save=False, show=True, prefix='plot_rv'):
        mod_rv = self.optimizier.filled_rv_model()

        pylab.plot(self.optimizier.rv_data[0], self.optimizier.rv_data[1], 'k.')
        pylab.plot(self.optimizier.photo_data[0], mod_rv, 'r')

        if save:
            pylab.savefig('./plots/{0}.png'.format(prefix))
        if show:
            pylab.show()

        pylab.close()