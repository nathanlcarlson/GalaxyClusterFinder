import sys

import numpy as np
# import scipy
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

class SDSSData:
    ''' A class for managing photometric and spectroscopic data '''

    def __init__(self, spec_csv, phot_csv, header_line=1, delimiter=','):
        # Load headers
        self.spec_h = {}
        self.phot_h = {}
        with open(spec_csv) as filep:
            for _ in range(header_line-1):
                next(filep)
            header = filep.readline().strip().split(',')
            for i in range(len(header)):
                self.spec_h[header[i]] = i
        with open(phot_csv) as filep:
            for _ in range(header_line-1):
                next(filep)
            header = filep.readline().strip().split(',')
            for i in range(len(header)):
                self.phot_h[header[i]] = i
        # Load data
        self.specd = np.genfromtxt(
            spec_csv,
            skip_header=header_line,
            delimiter=delimiter,
            missing_values=["null"],
            filling_values=-99.0
        )
        self.specd = self.specd[np.all(self.specd != -99.0, axis=1)]
        self.photd = np.genfromtxt(
            phot_csv,
            skip_header=header_line,
            delimiter=delimiter,
            missing_values=["null"],
            filling_values=-99.0
        )
        self.photd = self.photd[np.all(self.photd != -99.0, axis=1)]

        self.views = {}
    def spec(self, keys=None):
        ''' Get spec data with or without a key '''
        if not keys:
            return self.specd
        if type(keys) == type([]):
            a = []
            for key in keys:
                a.append(self.spec_h[key])
            return self.specd[:, a]
        return self.specd[:, self.spec_h[keys]]

    def phot(self, keys=None):
        ''' Get phot data with or without a key '''
        if not keys:
            return self.photd
        if type(keys) == type([]):
            a = []
            for key in keys:
                a.append(self.phot_h[key])
            return self.photd[:, a]
        return self.photd[:, self.phot_h[keys]]
    def add_view(self, name, view):
        ''' Submit a view '''
        self.views[name] = view
    def view(self, name):
        ''' Access a stored view '''
        return self.views[name]

def find_red_gals(data):
    ''' Find red spectroscopic galaxies in sample data '''
    # Get color
    data.add_view("g-r", data.spec('g') - data.spec('r'))
    color = data.view("g-r")

    # Get redshifts
    z = data.spec('redshift')

    # This will store the indecies of red galaxies
    reds = np.zeros(len(z)).astype(bool)

    # Width of reshift bins
    w = 0.05

    # Deepest redshift
    z_max = 0.35
    z_range = np.arange(0.05, z_max+0.01, w)
    for z_low, z_high in zip(z_range[:-1], z_range[1:]):
        # Indecies within z range
        in_z = (z > z_low)&(z < z_high)

        z_color = color[in_z]
        xbar = np.mean(z_color)
        s = np.std(z_color)
        z_color = z_color[(z_color < xbar+2.5*s)&(z_color > xbar-2.5*s)]
        X = z_color.reshape(-1, 1)

        # Model two peaks, blue and red
        init = np.array([[xbar], [xbar]])
        gmm = GaussianMixture(n_components=2, means_init=init).fit(X)

        # Red peak has a large mean color value of the two peaks
        red_gmm = np.argmax(gmm.means_)
        red_std = np.sqrt(gmm.covariances_[red_gmm][0][0])
        red_mean = gmm.means_[red_gmm][0]

        in_red = (color < red_mean+2*red_std)&(color > red_mean-2*red_std)
        reds[(in_red)&(in_z)] = True

    return reds

def reds_plot(data, reds):
    ''' Plot redshift vs color of red galaxies '''
    color = data.view("g-r")
    plt.plot(data.spec('redshift')[reds], color[reds], 'o')
    plt.show()

def main():
    ''' Main operations '''
    data = SDSSData(sys.argv[1], sys.argv[2])
    reds = find_red_gals(data)

    reds_plot(data, reds)

if __name__ == "__main__":
    main()
