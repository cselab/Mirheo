import numpy as np
from scipy.optimize import fsolve
import inspect

# ------ HELPER FUNCTIONS ------ #

def sq(x):
    return np.sqrt(x)

def calc_l0(A,Nv):
    return sq(A*4./(2*Nv-4)/sq(3))


def calc_kp(l0,lm,ks,m):
    return (6*ks*pow(l0,(m+1))*pow(lm,2)-9*ks*pow(l0,(m+2))*lm+4*ks*pow(l0,(m+3))) / (4*pow(lm,3)-8*l0*pow(lm,2)+4*pow(l0,2)*lm)

def calc_mu0(x0, l0, lm, ks, m):
    kp = calc_kp(l0, lm, ks, m)
    return sq(3)*ks/(4.*l0) * (x0/(2.*pow((1-x0),3)) - 1./(4.*pow((1-x0),2)) + 1./4) + sq(3)*kp*(m+1)/(4.*pow(l0,(m+1)))


def set_parameters(prms, gamma_in, eta_in, rho, lscale=1.0):
    
    # Max likelihood
    #
    #    B =  mu0 * R0*R0 / kb
    #    C = gammaC / gamma_in
    #    D = kb * rho / (eta_in*eta_in * R0)
    #    R0 = sqrt(Area/(4*pi))
    #
    x0=0.421125778395
    B=4.91149023426
    #C=7.66296686266
    C = 2
    D=0.0094841008474
    
    Nv = 498
    Area = 135.0
    R0 = np.sqrt(Area / (4*np.pi))
    
    mpow = 2.0
    
    kb = D * eta_in**2 * R0 / rho
    gammaC = C * (gamma_in)
    
    l0 = calc_l0(Area, Nv)
    lm = l0 / x0
    mu0 = B * kb / (R0**2)
    
    ks = fsolve(lambda ks : calc_mu0(x0, l0, lm, ks, mpow) - mu0, 0.1)[0]	

    
    prms.x0        = x0  
    prms.ka        = 49000.0
    prms.kb        = kb * lscale**2
    prms.kd        = 50000
    prms.kv        = 75000.0
    prms.gammaC    = gammaC * lscale
    prms.gammaT    = 0.0
    prms.kbT       = 0.0
    prms.mpow      = 2.0
    prms.theta     = 6.97
    prms.totArea   = Area * lscale**2
    prms.totVolume = 94 * lscale**3
    prms.ks        = ks
    prms.rnd       = False

    return prms


def params2dict(p):
    names = [ val[0] for val in inspect.getmembers(p.__class__, lambda m : hasattr(m, 'getter')) ]
    return { n : getattr(p, n) for n in names}

if __name__ == "__main__":
    
    class Dummy:
        def __init__(self):
            pass
    
    prms = Dummy()
    
    print(set_parameters(prms, 200, 200, 8).__dict__)
