__author__ = 'nmearl'

import os
import numpy as np
import pylab
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

plt.rc('font', family='serif')
plt.rc('font', serif='Times New Roman')


class Analyzer(object):
    def __init__(self, optimizer):
        self.optimizer = optimizer

        # burnin = int(0.5 * len(self.optimizer.chain))
        self.samples = self.optimizer.chain

        results = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]),
                      zip(*np.percentile(self.samples[:, 1:], [16, 50, 84],
                                         axis=0)))

        for i in range(len(self.optimizer.params.get_all(True))):
            param = self.optimizer.params.get_all(True)[i]
            param.quantile_value = results[i][0]
            param.upper_error = results[i][1]
            param.lower_error = results[i][2]

        params = self.optimizer.params
        # params.update(self.samples[
        # np.argmax(self.samples[:, 0]),
        # 1:])

        N = int(params.get("nbodies").value)

        self.masses = np.array([params.get('mass_{0}'.format(i))
                                for i in [1, 0, 2]])
        self.radii = np.array([params.get('radius_{0}'.format(i))
                               for i in [1, 0, 2]])
        self.fluxes = np.array([params.get('flux_{0}'.format(i))
                                for i in [1, 0, 2]])
        self.u1 = np.array(
            [params.get('u1_{0}'.format(i)) for i in [1, 0, 2]])
        self.u2 = np.array(
            [params.get('u2_{0}'.format(i)) for i in [1, 0, 2]])
        self.a = np.array(
            [params.get('a_{0}'.format(i)) for i in [1, 2]])
        self.e = np.array(
            [params.get('e_{0}'.format(i)) for i in [1, 2]])
        self.inc = np.array(
            [params.get('inc_{0}'.format(i)) for i in [1, 2]])
        self.om = np.array(
            [params.get('om_{0}'.format(i)) for i in [1, 2]])
        self.ln = np.array(
            [params.get('ln_{0}'.format(i)) for i in [1, 2]])
        self.ma = np.array(
            [params.get('ma_{0}'.format(i)) for i in [1, 2]])

    def report(self):
        redchisq = self.optimizer.redchisq()
        likelihood = 0.0  # np.max(self.samples[:, 0][self.samples[:, 0] !=
        # 0.0])

        print("Likelihood: {0}".format(likelihood))
        print("Reduced Chi-squared: {0}".format(redchisq))

        print("{0:12s} {1:12s} {2:12s} {3:12s}".format(
            'Name', 'Value', 'Err-', 'Err+'))

        j = 0
        with open("report.out", 'w') as f:
            for i in range(len(self.optimizer.params.get_all())):
                param = self.optimizer.params.get_all()[i]
                qval, quperr, qlowerr = param.get_real_quantile()

                if param.vary:
                    print("{0:12s} {1:12g} {2:12g} {3:12g}".format(
                        param.name, qval, quperr, qlowerr))
                    print(
                        "$\SI{{{0:e}}}{{}}\substack{{+\SI{{{1:e}}}{{}} \\ "
                        "-\SI{{{2:e}}}{{"
                        "}}}}$".format(
                            qval, quperr, qlowerr))

                    f.write("{0:12s} {1:12g} {2:12g} {3:12g}\n".format(
                        param.name, qval, quperr, qlowerr))

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

    def save_plot(self, method, param_list=None, prefix='hist'):
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
        for i in range(1, len(self.optimizer.params.get_all(True))):
            pylab.plot(self.samples[:, i], self.samples[:, 0], '+')
            pylab.show()

    def plot_histogram(self, show=False, save=False):
        carter_params = [(0.2413, 0.2127, 1.347), (.2318, .2543, 2.0254),
                         (3.26e-4, 2.24e-4, 1.0), (0.0, 0.0, 0.39),
                         (0.0, 0.0, 0.22),
                         (0.021986, 0.2495), (0.02234, 0.3043),
                         (96.907, 92.100), (89.52, 52.88), (8.012, 0.0),
                         (355.66, 19.87)]
        params = self.optimizer.params.get_all(True)

        titles = ["Star B", "Star C", "Star A"]

        gparams = [self.masses, self.radii, self.fluxes, self.u1, self.u2,
                   self.a, self.e, self.inc, self.om, self.ln, self.ma]
        si = 0

        for i in range(len(gparams)):
            gparam = gparams[i]
            cparam = carter_params[i]
            fig, axes = plt.subplots(len(gparam))

            if "mass" in gparam[0].name:
                par_name = "Mass (M$_\odot$)"
            elif "radius" in gparam[0].name:
                par_name = "Radius (R$_\odot$)"
            elif "flux" in gparam[0].name:
                par_name = "Flux"
            elif "u1" in gparam[0].name:
                par_name = "Primary Limb Darkening"
            elif "u2" in gparam[0].name:
                par_name = "Secondary Limb Darkening"
            elif "a_" in gparam[0].name[:2]:
                par_name = "Semi-major Axis (AU)"
            elif "e_" in gparam[0].name:
                par_name = "Eccentricity"
            elif "inc_" in gparam[0].name:
                par_name = "Inclination (deg)"
            elif "om_" in gparam[0].name:
                par_name = "Argument of Pericenter (deg)"
            elif "ln_" in gparam[0].name:
                par_name = "Longitude of Ascending Node (deg)"
            elif "ma_" in gparam[0].name:
                par_name = "Mean Anomaly (deg)"
            elif "gamma_t" == gparam[0].name:
                par_name = "TRES Correction (km/s)"
            elif "gamma_m" == gparam[0].name:
                par_name = "McDonald Correction (km/s)"
            else:
                par_name = ""

            fig.subplots_adjust(hspace=0.33)
            # fig.set_size_inches(8.5, 11)
            axes[-1].set_xlabel("{0}".format(par_name))

            for j in range(len(gparam)):
                param = gparam[j]
                cp = cparam[j]

                if any(x in param.name for x in
                       ["mass", "radius", "flux", "u1", "u2"]):
                    title = titles[j]
                else:
                    title = titles[j + 1]

                si += 1
                samples = self.samples[:, si]

                if "mass" in param.name:
                    samples /= 2.959122E-4
                elif "radius" in param.name:
                    samples *= 215.1
                elif ("inc" in param.name or "om" in param.name or
                              "ln" in param.name or "ma" in param.name):
                    samples = np.rad2deg(samples)
                elif "gamma" in param.name:
                    samples /= 5.775e-4

                qval, qupper, qlowerr = param.get_real_quantile()

                axes[j].set_title(title)  # , x=0.85, y=0.85)
                axes[j].set_ylabel("Count")

                axes[j].hist(samples, 1000, color="k", alpha=0.5,
                             histtype="step", facecolor='gray')
                axes[j].axvline(qval, color="r", lw=2)
                axes[j].axvline(qval + qupper, color='k', ls='-.')
                axes[j].axvline(qval - qlowerr, color='k', ls='-.')
                # axes[j].axvline(cp, ls='--')
                axes[j].set_xlim((qval - qlowerr) * 0.98,
                                 (qval + qupper) * 1.02)

            if save:
                plt.savefig("dist_{0}.png".format(gparam[0].name), dpi=300,
                            bbox_inches='tight')
            if show:
                plt.show()

            plt.close()

    def plot_flux(self, save=False, show=True, prefix='plot_flux'):
        mod_flux, mod_rv = self.optimizer.model()

        pylab.plot(self.optimizer.photo_data[0],
                   self.optimizer.photo_data[1], 'k+')
        pylab.plot(self.optimizer.photo_data[0],
                   mod_flux, 'r')

        if save:
            pylab.savefig('./plots/{0}.png'.format(prefix))
        if show:
            pylab.show()

        pylab.close()

    def plot_rv_full(self, show=False, save=False):
        mod_flux, mod_rv = self.optimizer.model()
        filled_x = np.linspace(50, 500, 1000)
        filled_rv = self.optimizer.filled_rv_model(
            filled_x)

        rv_x, rv_y, rv_e = self.optimizer.rv_data

        rv_e /= 5.775e-4
        rv_y = (rv_y / 5.775e-4)  # - 0.26 - 27.278
        mod_rv = (mod_rv / 5.775e-4)  # - 0.26 - 27.278
        filled_rv = (filled_rv / 5.775e-4)  # - 0.26 - 27.278

        print(mod_rv)
        # REMOVE THIS LATER
        trv_corr = self.optimizer.params.get("gamma_t").get_real()
        mrv_corr = self.optimizer.params.get("gamma_m").get_real()
        mod_rv[:10] += trv_corr
        mod_rv[10:] += mrv_corr
        filled_rv += (trv_corr + mrv_corr) * 0.5

        print(mod_rv)

        gs = gridspec.GridSpec(3, 1)
        fig = plt.figure()
        # fig.subplots_adjust(hspace=0.05)

        # fig.set_size_inches(11, 8.5)

        top_plot = plt.subplot(gs[0:2, 0])
        bottom_plot = plt.subplot(gs[2, 0])

        bottom_plot.set_xlabel('Time (BJD - 2,455,000)')
        bottom_plot.set_ylabel("Residuals")

        top_plot.set_ylabel('Radial Velocity of A (km/s)')

        plt.setp(top_plot.get_xticklabels(), visible=False)
        top_plot.set_xlim(45, 505)
        bottom_plot.set_xlim(45, 505)
        # bottom_plot.set_ylim(-0.7, 0.7)

        top_plot.plot(rv_x, rv_y, 'ko')
        # top_plot.plot(rv_x, mod_rv, 'bD')
        top_plot.plot(filled_x, filled_rv, 'r')

        bottom_plot.errorbar(rv_x[:10], rv_y[:10] - mod_rv[:10],
                             yerr=rv_e[:10], color='k', fmt='o')
        bottom_plot.errorbar(rv_x[10:], rv_y[10:] - mod_rv[10:],
                             yerr=rv_e[10:], color='k', fmt='o')
        bottom_plot.axhline(0.0, ls='--', color='k', alpha=0.5)

        if save:
            pylab.savefig("rv_full.png", dpi=150,
                          bbox_inches='tight')
        if show:
            plt.show()

        plt.close()

    def plot_mass_radius(self):
        names = ["Star B", "Star C", "Star A"]
        params = self.optimizer.params

        plt.subplot(2, 1, 1)
        for i in range(int(params.get("nbodies").value)):
            mass = params.get("mass_{0}".format(i)).get_real_quantile()
            radius = params.get("radius_{0}".format(i)).get_real_quantile()

            plt.errorbar(radius[0], mass[0],
                         xerr=[[radius[1], ], [radius[2], ]],
                         yerr=[[mass[1], ], [mass[2], ]], fmt='o',
                         label=names[i])

        plt.subplot(2, 3, 4)
        mass = params.get("mass_0").get_real_quantile()
        radius = params.get("radius_0").get_real_quantile()
        plt.errorbar(radius[0], mass[0], xerr=[[radius[1], ], [radius[2], ]],
                     yerr=[[mass[1], ], [mass[2], ]], fmt='o', label=names[i])

        plt.subplot(2, 3, 5)
        mass = params.get("mass_1").get_real_quantile()
        radius = params.get("radius_1").get_real_quantile()
        plt.errorbar(radius[0], mass[0], xerr=[[radius[1], ], [radius[2], ]],
                     yerr=[[mass[1], ], [mass[2], ]], fmt='o', label=names[i])

        plt.subplot(2, 3, 6)
        mass = params.get("mass_2").get_real_quantile()
        radius = params.get("radius_2").get_real_quantile()
        plt.errorbar(radius[0], mass[0], xerr=[[radius[1], ], [radius[2], ]],
                     yerr=[[mass[1], ], [mass[2], ]], fmt='o', label=names[i])

        plt.show()

    def plot_eclipse(self, t_start, period, show=False, save=False):
        mod_flux, mod_rv = self.optimizer.model()
        time = self.optimizer.photo_data[0]
        flux = self.optimizer.photo_data[1]

        ieclipse = -1
        for i in np.arange(t_start, time[-1], period * 4):
            ieclipse += 1
            gs = gridspec.GridSpec(3, 4)
            fig = plt.figure()
            fig.subplots_adjust(hspace=0.05, wspace=0.05)
            # fig.tight_layout()

            # fig.set_size_inches(11, 8.5)

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

            top_plots[0].set_ylabel("Normalized Flux")
            # bottom_plots[0].set_xlabel('Time (BJD - 2,455,000)')
            fig.text(0.5, 0.04, 'Time (BJD - 2,455,000)', ha='center',
                     va='center')
            bottom_plots[0].set_ylabel("Residuals")
            spacing = 0.5

            for j in range(4):
                top_plots[j].set_xlim(i + period * j - spacing,
                                      i + period * j + spacing)
                top_plots[j].set_ylim(0.968, 1.005)
                top_plots[j].plot(time, flux, 'k.')
                top_plots[j].plot(time[np.argsort(time)],
                                  mod_flux[np.argsort(time)], 'r')
                # top_plots[j].autoscale(tight=True)

                bottom_plots[j].set_xlim(i + period * j - spacing,
                                         i + period * j + spacing)
                bottom_plots[j].plot(time, flux - mod_flux, 'k.')
                bottom_plots[j].set_ylim(-0.004, 0.004)
                # bottom_plots[j].autoscale(tight=True)

            if save:
                plt.savefig("resid_{0}.png".format(ieclipse), dpi=150,
                            bbox_inches='tight')
            if show:
                plt.show()

            plt.close()