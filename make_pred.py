# working with a toy model assuming the gaussian profiles
# but can be easily extended to exotic ones as well!

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson

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
    def bm(self,m):
        "using eq 25 from ciotti and bertin 1999"
        ans = 2*m - 1/3 + 4/(405*m) + 46/(25515*m**2)
        return ans

    def sersic(self, I0, Re, m, x0, y0):
        "provides the functional form of profile"
        bm  = self.bm(m)
        ans = lambda x,y : I0 * np.exp(- bm * ( ((x-x0)**2 + (y-y0)**2)**0.5 / Re )**(1/m))
        return ans

class lenslight(galprofile):
    def __init__(self,nbins=1001, nRe=4):
        "odd because we are using simpsons"
        self.nbins  = nbins
        self.nRe    = nRe

    def src_profile(self, I0=1.0, Re=1, m=4, x0=0.0, y0=0.0):
        self.psrc    = self.sersic(I0, Re, m, x0, y0)
        xx = np.linspace(-self.nRe*Re + x0, self.nRe*Re + x0, self.nbins)
        yy = np.linspace(-self.nRe*Re + y0, self.nRe*Re + y0, self.nbins)
        self.x_grid, self.y_grid = np.meshgrid(xx,yy)
        return 0

    def lens_profile(self, I0=0.0, Re=1, m=4, x0=0.0, y0=0.0):
        self.plens = self.sersic(I0, Re, m, x0, y0)
        return 0

    def ell(self):
        "we use epsilon as the defination"
        zz = self.psrc(self.x_grid, self.y_grid) + self.plens(self.x_grid, self.y_grid)

        zz /=simpson(simpson(zz))
        xavg = simpson(simpson(self.x_grid * zz))
        yavg = simpson(simpson(self.y_grid * zz))

        q11 =  simpson(simpson((self.x_grid - xavg)**2 * zz))
        q22 =  simpson(simpson((self.y_grid - yavg)**2 * zz))

        q12 = simpson(simpson((self.x_grid - xavg) * (self.y_grid - yavg) * zz))

        e1 = (q11 - q22)/(q11 + q22)
        e2 = 2*q12/(q11 + q22)
        return e1, e2



if __name__ == "__main__":
    ax1 = plt.subplot(2,2,1)
    ax2 = plt.subplot(2,2,2)

    ll = lenslight(nbins=2001)
    ll.src_profile(I0=1, Re=1, m=4, x0=0.0, y0=0.0)
    for xx in np.logspace(1,3,100):
        for cnt,ii in enumerate([5,10,20]):
            ll.lens_profile(I0=ii, Re=3, m=4, x0=xx, y0=0.0)

            zz = ll.psrc(ll.x_grid, ll.y_grid)
            zz /=simpson(simpson(zz,axis=1))
            e1, e2 = ll.ell()
            ax1.plot(xx,-e1, '.', c='C%d'%(cnt))
            ax2.plot(xx,e2, '.', c='C%d'%(cnt))


    ax1.set_xscale('log')
    ax2.set_xscale('log')

    plt.savefig('test.png', dpi=300)

