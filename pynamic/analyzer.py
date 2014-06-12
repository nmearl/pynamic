__author__ = 'nmearl'

import os
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


class Analyzer(object):
    def __init__(self, optimizer, path="chain.txt"):
        self.optimizer = optimizer

        # burnin = int(0.5 * len(self.optimizer.chain))
        self.samples = self.optimizer.chain

        if path:
            self.samples = np.loadtxt(path, ndmin=2)

        self.results = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                           zip(*np.percentile(self.samples[:, 1:], [16, 50, 84],
                                              axis=0)))

    def report(self):
        mod_flux, mod_rv = self.optimizer.model()
        chisq = np.sum(((self.optimizer.photo_data[1] - mod_flux)
                        / self.optimizer.photo_data[2]) ** 2)
        deg = len(self.optimizer.params.get_flat(True))
        nu = self.optimizer.photo_data[1].size - 1.0 - deg
        redchisq = chisq / nu

        print("Reduced Chi-squared: {0}".format(redchisq))

        print("{0:12s} {1:12s} {2:12s} {3:12s}".format(
            'Name', 'Value', 'Err-', 'Err+'))

        j = 0
        with open("report.out", 'w') as f:
            for i in range(len(self.optimizer.params.all())):
                param = self.optimizer.params.all()[i]

                if "mass" in param.name:
                    lower_err = self.results[j][0] / 2.959122E-4
                    upper_err = self.results[j][1] / 2.959122E-4
                elif "radius" in param.name:
                    lower_err = self.results[j][0] * 215.1
                    upper_err = self.results[j][1] * 215.1
                elif ("inc" in param.name or "om" in param.name
                      or "ln" in param.name or "ma" in param.name):
                    lower_err = np.rad2deg(self.results[j][0])
                    upper_err = np.rad2deg(self.results[j][1])
                else:
                    lower_err = self.results[j][0]
                    upper_err = self.results[j][1]

                if param.vary:
                    print("{0:12s} {1:12g} {2:12g} {3:12g}".format(
                        param.name, lower_err, upper_err,
                        self.results[j][2]))

                    f.write("{0:12s} {1:12g} {2:12g} {3:12g}\n".format(
                        param.name, lower_err, upper_err,
                        self.results[j][2]))

                    j += 1
                else:
                    print("{0:12s} {1:12g}".format(param.name,
                                                   param.get_real()))

                    f.write("{0:12s} {1:12g}\n".format(param.name,
                                                       param.get_real()))

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
        if not os.path.exists("./plots"):
            os.mkdir("./plots")

        if method == 'histogram':
            self.histogram(save=True, show=False, param_list=param_list,
                           prefix=prefix)
        elif method == 'flux':
            self.plot_flux(save=True, show=False)
        elif method == 'rv':
            self.plot_rv(save=True, show=False)
        elif method == 'chi':
            self.chi(param_list=param_list, save=True, show=False)

    def chi(self, param_list=None, show=True, save=False):
        for i in range(1, len(self.optimizer.params.all(True))):
            pylab.plot(self.samples[:, i], self.samples[:, 0], '+')
            pylab.show()

    def histogram(self, param_list=None, save=False, show=True, prefix='hist'):
        for i in range(0, len(self.optimizer.params.all(True))):
            param = self.optimizer.params.all(True)[i]

            if param_list and param.name not in param_list:
                continue

            ax = pylab.subplot('111')#.format(res, j+1))
            ax.hist(self.samples[:, i], 100, color="k", alpha=0.5,
                    histtype="step")
            ax.axvline(self.results[i][0], color="r", lw=2)
            ax.axvline(self.results[i][0] + self.results[i][1],
                       color='k', alpha=0.6)
            ax.axvline(self.results[i][0] - self.results[i][2],
                       color='k', alpha=0.6)
            ax.set_title("{0:s} Parameter Distribution".format(param.name))
            ax.set_xlim(self.results[i][0] * 0.95, self.results[i][0] * 1.05)
            pylab.tight_layout()

            if save:
                pylab.savefig('./plots/{0}_{1:s}.png'.format(prefix,
                                                             param.name))
            if show:
                pylab.show()

            pylab.close()

    def step(self, param_list=None, save=False, show=True, prefix='step'):

        for i in range(1, len(self.optimizer.params.all(True))):
            pylab.figure()

            param = self.optimizer.params.all(True)[i]

            if param_list and param.name not in param_list:
                continue

            ax = pylab.subplot('111')
            ax.plot(range(len(self.samples)), self.samples[:, i],
                    color="k", alpha=0.5)
            ax.axhline(self.results[i][0], color="r", lw=2)
            ax.axhline(self.results[i][0] + self.results[i][1],
                       color='k', alpha=0.6)
            ax.axhline(self.results[i][0] - self.results[i][2],
                       color='k', alpha=0.6)
            ax.set_title("{0:s} Step".format(param.name))
            # ax.set_xlim(results[k][0] * 0.95, results[k][0] * 1.05)
            pylab.tight_layout()

            if save:
                pylab.savefig('./plots/{0}_{1:s}.png'.format(prefix,
                                                             param.name))
            if show:
                pylab.show()

            pylab.close()

    def plot_flux(self, save=False, show=True, prefix='plot_flux'):
        mod_flux, mod_rv = self.optimizer.model()

        pylab.plot(self.optimizer.photo_data[0],
                   self.optimizer.photo_data[1], 'k+')
        pylab.plot(self.optimizer.photo_data[0], mod_flux, 'r')

        if save:
            pylab.savefig('./plots/{0}.png'.format(prefix))
        if show:
            pylab.show()

        pylab.close()

    def plot_rv(self, save=False, show=True, prefix='plot_rv'):
        mod_rv = self.optimizer.filled_rv_model()

        pylab.plot(self.optimizer.rv_data[0], self.optimizer.rv_data[1], 'k.')
        pylab.plot(self.optimizer.photo_data[0], mod_rv, 'r')

        if save:
            pylab.savefig('./plots/{0}.png'.format(prefix))
        if show:
            pylab.show()

        pylab.close()

    def plot_eclipse(self, t_start, period):
        mod_flux, mod_rv = self.optimizer.model()
        time = self.optimizer.photo_data[0]
        flux = self.optimizer.photo_data[1]

        for i in np.arange(t_start, time[-1], period * 4):
            gs = gridspec.GridSpec(3, 4)
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.05, wspace=0.05)
            # fig.tight_layout()

            fig.set_size_inches(11, 8.5)

            fig.text(0.5, 0.04, 'Time (BJD - 2,455,000)', ha='center',
                     va='center')
            fig.text(0.06, 0.5, 'Normalized Flux', ha='center', va='center',
                     rotation='vertical')

            top_plots, bottom_plots = [], []

            for j in range(4):
                if j > 0:
                    top_plots.append(plt.subplot(gs[0:2, j],
                                                 sharey=top_plots[0]))
                    plt.setp(top_plots[j].get_yticklabels(), visible=False)
                    bottom_plots.append(plt.subplot(gs[2, j],
                                                    sharey=bottom_plots[0]))
                    bottom_plots[j].get_yaxis().set_visible(False)
                else:
                    top_plots.append(plt.subplot(gs[0:2, j]))
                    bottom_plots.append(plt.subplot(gs[2, j]))

                plt.setp(top_plots[j].get_xticklabels(), visible=False)
                bottom_plots[j].yaxis.set_major_locator(MaxNLocator(4))
                bottom_plots[j].xaxis.set_major_locator(MaxNLocator(4))
                top_plots[j].xaxis.set_major_locator(MaxNLocator(4))
                # bottom_plots[j].get_xaxis().get_major_formatter(
                # ).set_powerlimits((0, 0))
                # bottom_plots[j].ticklabel_format(style='sci',
                # scilimits=(0, 0), axis='x')
                top_plots[j].get_xaxis().get_major_formatter().set_useOffset(
                    False)
                bottom_plots[j].get_xaxis().get_major_formatter().set_useOffset(
                    False)

                plt.setp(bottom_plots[j].xaxis.get_majorticklabels(),
                         rotation=15)

            spacing = 0.5

            for j in range(4):
                top_plots[j].set_xlim(i + period * j - spacing,
                                      i + period * j + spacing)
                top_plots[j].set_ylim(0.968, 1.005)
                top_plots[j].plot(time, flux, 'k.')
                top_plots[j].plot(time, mod_flux, 'r')
                # top_plots[j].autoscale(tight=True)

                bottom_plots[j].set_xlim(i + period * j - spacing,
                                         i + period * j + spacing)
                bottom_plots[j].plot(time, flux - mod_flux, 'k.')
                bottom_plots[j].set_ylim(-0.004, 0.004)
                # bottom_plots[j].autoscale(tight=True)

            # plt.show()

            pylab.savefig("resid_{0}.png".format(i), dpi=300,
                          bbox_inches='tight')