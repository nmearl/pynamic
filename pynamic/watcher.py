__author__ = 'nmearl'

import threading
import numpy as np


class Watcher(threading.Thread):

    def __init__(self, optimizer, verbose=False):
        threading.Thread.__init__(self)
        self.optimizer = optimizer
        self.verbose = verbose
        self.running = True
        self.maxlnp = 0.0

    def stop(self):
        self.running = False


class Progress(Watcher):

    def run(self):
        import time

        while self.running:
            time.sleep(10.0)

            if not self.running:
                break

            if self.optimizer.maxlnp != self.maxlnp:
                self.maxlnp = self.optimizer.maxlnp
                self.iterprint()
                # self.optimizer.save()

    def iterprint(self):
        nbodies = int(self.optimizer.params.get('nbodies').value)

        masses = np.array([self.optimizer.params.get('mass_{0}'.format(i)).value for i in range(nbodies)])
        radii = np.array([self.optimizer.params.get('radius_{0}'.format(i)).value for i in range(nbodies)])
        fluxes = np.array([self.optimizer.params.get('flux_{0}'.format(i)).value for i in range(nbodies)])
        u1 = np.array([self.optimizer.params.get('u1_{0}'.format(i)).value for i in range(nbodies)])
        u2 = np.array([self.optimizer.params.get('u2_{0}'.format(i)).value for i in range(nbodies)])

        a = np.array([self.optimizer.params.get('a_{0}'.format(i)).value for i in range(1, nbodies)])
        e = np.array([self.optimizer.params.get('e_{0}'.format(i)).value for i in range(1, nbodies)])
        inc = np.array([self.optimizer.params.get('inc_{0}'.format(i)).value for i in range(1, nbodies)])
        om = np.array([self.optimizer.params.get('om_{0}'.format(i)).value for i in range(1, nbodies)])
        ln = np.array([self.optimizer.params.get('ln_{0}'.format(i)).value for i in range(1, nbodies)])
        ma = np.array([self.optimizer.params.get('ma_{0}'.format(i)).value for i in range(1, nbodies)])

        print('=' * 83)
        print('Likelihood: {0} | Red. Chisq: {1}'.format(
            self.maxlnp, self.optimizer.redchisq))#, percomp * 100, time.strftime('%H:%M:%S', time.gmtime(tleft))))
        print('-' * 83)
        print('System parameters')
        print('-' * 83)
        print(
            '{0:11s} {1:11s} {2:11s} {3:11s} {4:11s} {5:11s} '.format(
                'Body', 'Mass', 'Radius', 'Flux', 'u1', 'u2'
            )
        )

        for i in range(nbodies):
            print(
                '{0:11s} {1:1.5e} {2:1.5e} {3:1.5e} {4:1.5e} {5:1.5e}'.format(
                    str(i + 1), masses[i], radii[i], fluxes[i], u1[i], u2[i]
                )
            )

        print('-' * 83)
        print('Keplerian parameters')
        print('-' * 83)

        print(
            '{0:11s} {1:11s} {2:11s} {3:11s} {4:11s} {5:11s} {6:11s}'.format(
                'Body', 'a', 'e', 'inc', 'om', 'ln', 'ma'
            )
        )

        for i in range(nbodies - 1):
            print(
                '{0:11s} {1:1.5e} {2:1.5e} {3:1.5e} {4:1.5e} {5:1.5e} {6:1.5e}'.format(
                    str(i + 2), a[i], e[i], inc[i], om[i], ln[i], ma[i]
                )
            )

        print('')