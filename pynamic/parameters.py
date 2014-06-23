__author__ = 'nmearl'

from collections import OrderedDict
import numpy as np


class Parameters(object):

    def __init__(self, input_file):
        self.odict = OrderedDict()
        self._read_input(input_file)

        if "ferr_frac" not in self.odict.keys():
            self.add("ferr_frac", 0.0, 0.0, 1.0, False)

    def _read_input(self, input_file):
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip().split()

                self.add(name=line[0],
                         initvalue=float(line[1])
                         if line[0] is not "nbodies" else int(line[1]),
                         min=float(line[2]) if "inf" not in line[
                             2] else -np.inf,
                         max=float(line[3]) if "inf" not in line[3] else np.inf,
                         vary=bool(int(line[4])))

    def add(self, name, initvalue, min=-np.inf, max=np.inf, vary=True):
        param = Parameter(name, initvalue, min, max, vary)
        self.odict[name] = param

    def get(self, name='', index=0):
        if name:
            return self.odict[name]

        return self.odict.items()[index][1]

    def get_flat(self, can_vary=False, quantile=False):
        if can_vary and quantile:
            return np.array([x.quantile_value
                             for x in self.odict.values() if x.vary])
        elif can_vary:
            return np.array([x.value for x in self.odict.values() if x.vary])

        return np.array([x.value for x in self.odict.values()])

    def get_all(self, can_vary=False):
        if can_vary:
            return [x for x in self.odict.values() if x.vary]

        return self.odict.values()

    def get_bounds(self, can_vary=False):
        if can_vary:
            return [(x.min, x.max) for x in self.odict.values() if x.vary]

        return [(x.min, x.max) for x in self.odict.values()]

    def update(self, theta):
        for i in range(len(theta)):
            p = self.get_all(True)[i]
            p.value = theta[i]

    def save(self):
        with open("current.out", "w") as f:
            for param in self.odict.values():
                f.write("{0:12s} {1} {2:12g} {3:12g} {4:4d}\n".format(
                    param.name, param.value, param.min, param.max, param.vary
                ))


class Parameter(object):

    def __init__(self, name, value, min, max, vary):
        self.name = name
        self.initvalue = value
        self.value = value
        self.min = min
        self.max = max
        self.vary = vary
        self.upper_error = 0.0
        self.lower_error = 0.0
        self.quantile_value = 0.0

    def get_real(self):
        if "mass" in self.name:
            return self.value / 2.959122E-4
        elif "radius" in self.name:
            return self.value * 215.1
        elif "gamma" in self.name:
            return self.value / 5.775e-4
        elif ("inc_" in self.name or "om_" in self.name or "ln_" in self.name
              or "ma_" in self.name[:3]):
            return np.rad2deg(self.value)
        else:
            return self.value

    def get_real_quantile(self):
        if "mass" in self.name:
            return np.array([self.quantile_value,
                             self.upper_error, self.lower_error]) / 2.959122E-4
        elif "radius" in self.name:
            return np.array([self.quantile_value,
                             self.upper_error, self.lower_error]) * 215.1
        elif "gamma" in self.name:
            return np.array([self.quantile_value,
                             self.upper_error, self.lower_error]) / 5.775e-4
        elif ("inc" in self.name or "om" in self.name or "ln" in self.name
              or "ma_" in self.name[:3]):
            return np.rad2deg(np.array([self.quantile_value,
                                        self.upper_error, self.lower_error]))
        else:
            return np.array([self.quantile_value,
                             self.upper_error, self.lower_error])


    def __repr__(self):
        return "{0:12} {1:12g} {2:12g} {3:12g} {4:6}".format(self.name,
                                                             self.value,
                                                             self.min,
                                                             self.max,
                                                             self.vary)