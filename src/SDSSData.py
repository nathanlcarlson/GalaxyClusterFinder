import sys

import numpy as np
from astropy.cosmology import Planck13
import astropy.units as AU
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from sklearn.mixture import GaussianMixture
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import RectBivariateSpline
from scipy import optimize

COSMO = Planck13
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


# TODO Backgroud esitmates!!
def G(c, c_z, z, sig=0.05):
    a = 1/(np.sqrt(2*np.pi)*sig)
    b = -1*np.power(c - c_z(z), 2)/(2*sig*sig)
    return a*np.exp(b)

def m_star_z(z):

    return 12.27 + 62.36*z - 289.79*np.power(z, 2) + 729.69*np.power(z, 3) - 709.42*np.power(z, 4)
    # return 12.27 + 62.36*z - 289.79*z**2 + 729.69*z**3 − 709.42*z**4

def phi(m, z, m_star=None, a=0.8):
    if not m_star:
        m_star = m_star_z(z)
    return np.power(10, -0.4*(m-m_star)*(a+1))*np.exp(-1*np.power(10, -0.4*(m-m_star)))

def sigma(R, R_s=0.15, R_core=0.1):
    if R < R_core:
        return sigma(R_core)
    q = R/R_s
    a = 1/(q*q-1)
    if q < 1:
        x = np.sqrt(-1*(q-1)/(q+1))
        b = 1 - 2/np.sqrt(-1*(q*q-1))*np.arctanh(x)
    if q >= 1:
        x = np.sqrt((q-1)/(q+1))
        b = 1 - 2/np.sqrt(q*q-1)*np.arctan(x)
    return a*b
def get_red_model(data, reds):

    z = data.spec('redshift')[reds]
    color = data.view("g-r")[reds]

    z, unique_index = np.unique(z, return_index=True)
    color = color[unique_index]
    x = z[z.argsort()]
    y = color[z.argsort()]

    f = UnivariateSpline(x, y)
    # xnew = np.linspace(min(x), max(x), 1000)
    # plt.plot(x, y, 'o', xnew, f(xnew), '-')
    # plt.legend(['data', 'linear', 'cubic'], loc='best')
    # plt.show()
    return f

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

def background_est(data):

    specgr = data.spec("g") - data.spec("r")
    photgr = data.phot("g") - data.phot("r")
    alli = np.concatenate((data.spec("ci"), data.phot("ci")))
    allgr = np.concatenate((specgr, photgr))

    gr_edges = np.linspace(-1, 3, 41)
    i_edges = np.linspace(12, 22, 101)

    H, xedges, yedges = np.histogram2d(alli, allgr, bins=(i_edges, gr_edges))
    min_dec = min((min(data.spec("dec")), min(data.phot("dec"))))
    max_dec = max((max(data.spec("dec")), max(data.phot("dec"))))
    min_ra = min((min(data.spec("ra")), min(data.phot("ra"))))
    max_ra = max((max(data.spec("ra")), max(data.phot("ra"))))
    A = (max_dec-min_dec)*(max_ra-min_ra)
    H = H/0.1/0.1/A
    x = [(a+b)/2 for a, b in zip(xedges[:-1], xedges[1:])]
    y = [(a+b)/2 for a, b in zip(yedges[:-1], yedges[1:])]
    # points = np.mgrid(x, y)
    # values = np.ravel(H)

    return RectBivariateSpline(x, y, H)

def density(S_R, p_m, g_c):

    return S_R * p_m * g_c

def distance(t1, p1, t2, p2, mpc):
    a = 2*mpc*mpc*(1-(np.sin(t1)*np.sin(t2)*np.cos(p1-p2)+np.cos(t1)*np.cos(t2)))
    return np.sqrt(a)

def sigma_normalization(R_c=0.9):
    p = np.log(R_c)
    a = (
        1.6517 - 0.5479*p +
        0.1382*pow(p, 2) - 0.0719*pow(p, 3) -
        0.01582*pow(p, 4) - 0.00085499*pow(p, 5)
    )

    return np.exp(a)

def phi_normalization(z):
    m_star = m_star_z(z)
    m_lim = m_star + 1.75
    I = integrate.quad(phi, 0, m_lim, args=(z))[0]
    return 1/I

def func(x, u, b):
    u = x*u
    return x - np.sum(u/(u+b))

def main():
    ''' Main operations '''
    data = SDSSData(sys.argv[1], sys.argv[2])
    reds = find_red_gals(data)
    model = get_red_model(data, reds)

    bkg = background_est(data)
    vsigma = np.vectorize(sigma)
    vphi = np.vectorize(phi)
    vG = np.vectorize(G)
    vDist = np.vectorize(distance)

    spec_color = data.spec("g") - data.spec("r")
    phot_color = data.phot("g") - data.phot("r")
    spec_theta = data.spec("ra")*np.pi/180
    phot_theta = data.phot("ra")*np.pi/180
    spec_phi = (90.0-data.spec("dec"))*np.pi/180
    phot_phi = (90.0-data.phot("dec"))*np.pi/180

    color = np.concatenate((spec_color, phot_color))
    theta = np.concatenate((spec_theta, phot_theta))
    phi_s = np.concatenate((spec_phi, phot_phi))
    mags = np.concatenate((data.spec("ci"), data.phot("ci")))
    R_0 = sigma_normalization()

    print("Use seeds...")
    for t, p, c, z in zip(spec_theta[reds], spec_phi[reds], spec_color[reds], data.spec("redshift")[reds]):
        valids = np.zeros(len(color)).astype(bool)
        valids[mags<21.0] = True
        valids[(color < model(z)+2*0.05)&(color > model(z)-2*0.05)] = True
        if not np.any(valids):
            continue
        mpc = float(COSMO.luminosity_distance(z)/pow((z+1), 2)/AU.Mpc)
        Rs = vDist(t, p, theta[valids], phi_s[valids], mpc)
        validR =  Rs < .5
        valids[valids] = validR
        if not np.any(valids):
            continue

        Rs = Rs[validR]
        S_R = 2*np.pi*R_0*vsigma(Rs)
        # print(phi_normalization(z))
        p_m = phi_normalization(z)*vphi(mags[valids], z)
        G_c = vG(color[valids], model, z)
        # print("SigmaR", S_R)
        # print("PhiM", p_m)
        # print("Gc", G_c)
        rho = density(S_R, p_m, G_c)

        bkg_density = []
        for i in range(len(mags[valids])):
            bkg_density.append(bkg(mags[valids][i], color[valids][i])[0][0])
        bkg_density = np.array(bkg_density)
        bkg_density *= 2*np.pi*Rs
        # print("u", rho)
        # print("bkg", bkg_density)
        sol = optimize.root(func, 100, args=(rho, bkg_density))
        if sol.x > 10:
            print("z: {:01.6f}\nlambda: {:03.4f}".format(z, sol.x[0]))


    # rs = np.linspace(0, 1, 1000)
    # plt.plot(rs, vsigma(rs),'o')
    # plt.show()
    # ms = np.linspace(14, 22, 1000)
    # plt.plot(ms, vphi(ms, z=0.2),'o')
    # plt.show()
    # plt.hist(data.view("g-r")[reds])
    # reds_plot(data, reds)

    # cs = np.linspace(1.0, 2.0, 1000)
    # plt.plot(cs, vG(cs, model, 0.2), 'o')
    # plt.show()
if __name__ == "__main__":
    main()
