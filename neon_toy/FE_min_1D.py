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
m = 20.1797*utokg # kg
sigma = 2.6 # Angstrom
epsilon = 1.6*1000/na*4*1e20 # kg Ang^2 / s^2 = 1.6 kJ/mol * 4
beta = 1/kb/T/1e20
#print epsilon, beta

boxlen = 100.0 # Angstrom
nrhob = pow(0.033,1./3.)
ngrid = 500
niters = 10
dx = boxlen / ngrid
#dv = dx*dx*dx
rhob = nrhob
totaldensity = nrhob*boxlen
x = np.arange(0,boxlen,dx)

bhT = 0.01
ftol = 0.00001
gtol = 0.0001
bnd = [(0.000,1.e99)]*(ngrid)
stepsize = 0.0005


lmda = h/np.sqrt(2*pi*m*kb*T)*1e10 # Angstrom
#lmda = pow(lmda,3)
lrcutoff = pow(2,1./6.)*sigma
hrcutoff = 10.0
lcutoff = int(np.ceil(lrcutoff/dx))
hcutoff = int(np.floor(hrcutoff/dx))

def LJ(r):
    term = pow(sigma/r,6)
    return epsilon*(term*term-term)

ljr = np.zeros(hcutoff,dtype=float)
for i in range(lcutoff):
    ljr[i] = LJ(lrcutoff) - LJ(hcutoff)
for i in range(lcutoff,hcutoff):
    ljr[i] = LJ(i*dx) - LJ(hcutoff)

#plt.plot(np.arange(0.0,hrcutoff,dx),ljr)
#plt.show()
#exit()

fidk = dx/beta

rho = np.zeros(ngrid,dtype=float)
rho.fill(rhob)
@jit#(parallel=True)
def totalfe(rho):
    global ljr, ngrid
    ratio = np.sum(rho)*dx/totaldensity
    rho /= ratio
    FLJ = 0.0
    for x in range(ngrid):
        term = 0.0
        for i in range(1,hcutoff):
            term += ljr[i]*rho[x+i-ngrid]
        FLJ += term * rho[x]
    FID = 0.0
    for x in range(ngrid):
        if rho[x] > 0.0:
            FID += rho[x]*(np.log(lmda*rho[x])-1)
#    print('%.3f %.3f' %(FID*fidk, FLJ*dx))
    return FID*fidk + FLJ*dx
#print totalfe(rho)

#exit()

class MyTakeStep(object):
    def __init__(self, stepsize=0.001):
        self.stepsize = stepsize
    def __call__(self, x):
        s = self.stepsize*0.9
        x += np.random.uniform(-s,s,ngrid)
        return x


res = optimize.basinhopping(totalfe,rho,T=bhT
#    ,stepsize=MyTakeStep()
    ,niter_success=niters
    ,minimizer_kwargs={
    'method':'L-BFGS-B',
    'bounds':bnd,
    'options':{
    'gtol':gtol,
    'ftol':ftol,
    'disp': True}})

np.savetxt('rho.dat',(x,res.x))

data = np.loadtxt('rho.dat')
r = data[0]
rho = data[1]
plt.plot(r,rho)
plt.show()
plt.close()

grcut = int(np.ceil(10.0/dx))
nr = np.zeros(grcut,dtype=int)
gr = np.zeros(grcut,dtype=float)
for x in range(0,ngrid):
    for i in range(1,grcut):
        gr[i] += rho[x]*rho[x+i-ngrid]

plt.plot(np.arange(dx,10.0,dx),gr[1:])
plt.show()
plt.close()

