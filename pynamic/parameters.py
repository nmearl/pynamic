__author__ = 'nmearl'

from collections import OrderedDict
from numpy import inf, array


class Parameters(object):

    def __init__(self, input_file):
        self.odict = OrderedDict()
        self._read_input(input_file)

    def _read_input(self, input_file):
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip().split()

                self.add(name=line[0],
                         initvalue=float(line[1]) if line[1] is not "nbodies" else int(line[1]),
                         min=float(line[2]) if "inf" not in line[2] else -inf,
                         max=float(line[3]) if "inf" not in line[3] else inf,
                         vary=bool(int(line[4])))

    def add(self, name, initvalue, min=-inf, max=inf, vary=True):
        param = Parameter(name, initvalue, min, max, vary)
        self.odict[name] = param

    def get(self, name='', index=0):
        if name:
            return self.odict[name]

        return self.odict.items()[index][1]

    def get_flat(self, can_vary=False):
        if can_vary:
            return array([x.value for x in self.odict.values() if x.vary is True])

        return array([x.value for x in self.odict.values()])

    def all(self, can_vary=False):
        if can_vary:
            return [x for x in self.odict.values() if x.vary is True]

        return self.odict.values()

    def get_bounds(self, can_vary=False):
        if can_vary:
            return [(x.min, x.max) for x in self.odict.values() if x.vary is True]

        return [(x.min, x.max) for x in self.odict.values()]

    def update_parameters(self, theta):
        for i in range(len(theta)):
            p = self.all(True)[i]
            p.value = theta[i]


class Parameter(object):

    def __init__(self, name, value, min, max, vary):
        self.name = name
        self.initvalue = value
        self.value = value
        self.min = min
        self.max = max
        self.vary = vary

    def __repr__(self):
        return "{0:12} {1:12g} {2:12g} {3:12g} {4:6}".format(self.name, self.value, self.min, self.max, self.vary)