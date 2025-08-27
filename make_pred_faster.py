import numpy as np
import matplotlib.pyplot as plt

def get_et(lra, ldec, sra, sdec, se1, se2):
    lra  = lra*np.pi/180
    ldec = ldec*np.pi/180
    sra  = sra*np.pi/180
    sdec = sdec*np.pi/180

    c_theta = np.cos(ldec)*np.cos(sdec)*np.cos(lra - sra) + np.sin(ldec)*np.sin(sdec)
    s_theta = np.sqrt(1-c_theta**2)

    # phi to get the compute the tangential shear
    c_phi   = np.cos(ldec)*np.sin(sra - lra)*1.0/s_theta
    s_phi   = (-np.sin(ldec)*np.cos(sdec) + np.cos(ldec)*np.cos(sra - lra)*np.sin(sdec))*1.0/s_theta
    # tangential shear
    e_t     = - se1*(2*c_phi**2 -1) - se2*(2*c_phi * s_phi)

    return e_t

class galprofile():
    def bm(self, m):
        "using eq 25 from ciotti and bertin 1999"
        ans = 2*m - 1/3 + 4/(405*m) + 46/(25515*m**2)
        return ans

    def sersic(self, I0, Re, m, x0, y0):
        "provides the functional form of profile"
        bm  = self.bm(m)
        ans = lambda x, y: I0 * np.exp(-bm * (np.sqrt((x-x0)**2 + (y-y0)**2) / Re)**(1/m))
        return ans

class lenslight(galprofile):
    def __init__(self, nbins=1001, nRe=4):
        "odd because we are using trapz integration"
        self.nbins = nbins
        self.nRe = nRe
        # Pre-calculate dx and dy for integration
        self.dx = None
        self.dy = None

    def src_profile(self, I0=1.0, Re=1, m=4, x0=0.0, y0=0.0):
        self.psrc = self.sersic(I0, Re, m, x0, y0)
        xx = np.linspace(-self.nRe*Re + x0, self.nRe*Re + x0, self.nbins)
        yy = np.linspace(-self.nRe*Re + y0, self.nRe*Re + y0, self.nbins)
        self.x_grid, self.y_grid = np.meshgrid(xx, yy)
        # Store integration steps
        self.dx = xx[1] - xx[0]
        self.dy = yy[1] - yy[0]
        return 0

    def lens_profile(self, I0=0.0, Re=1, m=4, x0=0.0, y0=0.0):
        self.plens = self.sersic(I0, Re, m, x0, y0)
        return 0

    def ell(self):
        "we use epsilon as the definition - optimized version"
        # Calculate profiles once
        psrc_vals = self.psrc(self.x_grid, self.y_grid)
        plens_vals = self.plens(self.x_grid, self.y_grid)
        zz = psrc_vals + plens_vals

        # Use trapz instead of simpson - faster and sufficient accuracy
        total = np.trapz(np.trapz(zz, axis=1), dx=self.dy) * self.dx
        zz_norm = zz / total

        # Vectorized calculations
        xavg = np.trapz(np.trapz(self.x_grid * zz_norm, axis=1), dx=self.dy) * self.dx
        yavg = np.trapz(np.trapz(self.y_grid * zz_norm, axis=1), dx=self.dy) * self.dx

        # Pre-calculate differences
        x_diff = self.x_grid - xavg
        y_diff = self.y_grid - yavg

        q11 = np.trapz(np.trapz(x_diff**2 * zz_norm, axis=1), dx=self.dy) * self.dx
        q22 = np.trapz(np.trapz(y_diff**2 * zz_norm, axis=1), dx=self.dy) * self.dx
        q12 = np.trapz(np.trapz(x_diff * y_diff * zz_norm, axis=1), dx=self.dy) * self.dx

        q_sum = q11 + q22
        e1 = (q11 - q22) / q_sum
        e2 = 2 * q12 / q_sum

        return e1, e2

if __name__ == "__main__":
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)

    # Reduced grid size for better performance while maintaining accuracy
    ll = lenslight(nbins=501)  # Reduced from 2001
    ll.src_profile(I0=1, Re=1, m=4, x0=0.0, y0=0.0)

    # Pre-compute source profile normalization (only needs to be done once)
    zz_src = ll.psrc(ll.x_grid, ll.y_grid)
    total_src = np.trapz(np.trapz(zz_src, axis=1), dx=ll.dy) * ll.dx

    xx_vals = np.logspace(1, 3, 100)
    ii_vals = [5, 10, 20]

    for cnt, ii in enumerate(ii_vals):
        e1_vals = []
        e2_vals = []

        for xx in xx_vals:
            ll.lens_profile(I0=ii, Re=3, m=4, x0=xx, y0=0.0)
            e1, e2 = ll.ell()
            e1_vals.append(-e1)
            e2_vals.append(e2)

        ax1.plot(xx_vals, e1_vals, '.', c='C%d' % cnt)
        ax2.plot(xx_vals, e2_vals, '.', c='C%d' % cnt)

    ax1.set_xscale('log')
    ax2.set_xscale('log')

