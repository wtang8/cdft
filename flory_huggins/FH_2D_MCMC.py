import numpy as np
from scipy import optimize
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from numba import jit

utokg = 1.660539040e-27
h = 6.626070150e-34 # kg m^2 / s
pi = 3.14159265359
kb = 1.38064852e-23 # kg m^2 / s^2 K
na = 6.022140857e23 # Avogadro's number
T = 35.05 # K
beta = 1/kb/T/1e20

boxlen = 1000.0 # Angstrom
ngrid = 500
niters = 10
dx = boxlen / ngrid
x = np.arange(0,boxlen,dx)

rhob = 0.1
nsp = 2
rho = np.zeros((nsp*ngrid*ngrid),dtype=float)
rho.fill(rhob)
xpol = np.array([50,50])
chi = np.matrix('.2 .8; .8 .2')
totaldensity = rhob*boxlen

bhT = 0.01
ftol = 0.00001
gtol = 0.0001

lbnd = 0.000
ubnd = 1.e99
bnd = [(0.000,1.e99)]*(nsp*ngrid)
stepsize = rhob/20

#lmda = h/np.sqrt(2*pi*m*kb*T)*1e10 # Angstrom

#print np.sum(rho,axis=0)
normf = dx/totaldensity
@jit#(parallel=True)
def totalfe(rho):
    global ljr, ngrid
    rho = np.array(list(zip(rho[:ngrid],rho[ngrid:2*ngrid]))) 
    ratio = np.sum(rho,axis=0)*normf
    rho /= ratio
    fent = 0.0
    for i in range(nsp):
        for x in range(ngrid):
            if rho[x][i] > 0.0:
                fent += rho[x][i]/xpol[i]*np.log(rho[x][i])
    fmix = 0.0
    for x in range(ngrid):
        fmix += rho[x].dot(chi.dot(rho[x]))
#    print('%.3f %.3f'% (fent,fmix))
    return fent + fmix
print totalfe(rho)
#exit()

res = optimize.basinhopping(totalfe,rho,T=bhT
#    ,stepsize=stepsize
    ,niter_success=niters
    ,minimizer_kwargs={
    'method':'L-BFGS-B',
    'bounds':bnd,
    'options':{
    'gtol':gtol,
    'ftol':ftol,
    'disp': True}})

np.savetxt('rho.dat',res.x)

data = np.loadtxt('rho.dat')
rho = data
plt.plot(x,rho[:ngrid],color='r')
plt.plot(x,rho[ngrid:2*ngrid],color='b')
plt.show()
plt.close()

