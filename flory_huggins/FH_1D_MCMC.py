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
#beta = 1/kb/T/1e20
beta = 10

boxlen = 1000.0 # Angstrom
ngrid = 500
dx = boxlen / ngrid
x = np.arange(0,boxlen,dx)

rhob = 0.1
nsp = 2
rho = np.zeros(nsp*ngrid,dtype=float)
rho[0:int(np.floor(ngrid/2))] = rhob
rho[-int(np.floor(ngrid/2)):] = rhob
#rho.fill(rhob)
xpol = np.array([50,50])
chi = np.matrix('.2 .8; .8 .2')
#print chi
totaldensity = rhob*ngrid

mciter = 1000
mcstep = rhob/20

lbnd = 0.000
ubnd = 1.e99

#lmda = h/np.sqrt(2*pi*m*kb*T)*1e10 # Angstrom
#print np.sum(rho,axis=0)

@jit#(parallel=True)
def totalfe(rho):
    global ljr, ngrid
    fent = 0.0
    for i in range(nsp):
        for x in range(ngrid):
            y = i*ngrid+x
            if rho[y] > 0.0:
                fent += rho[y]/xpol[i]*np.log(rho[y])
    fmix = 0.0
    v = np.zeros(nsp,dtype=float)
    for x in range(ngrid):
        v[0] = rho[x]
        v[1] = rho[ngrid+x] 
        fmix += v.dot(chi.dot(v))
#    print('%.3f %.3f'% (fent,fmix))
    return fent + fmix

feovertime = []
def TakeStep(rho,p0):
    x = rho
    rho += np.random.uniform(-mcstep,mcstep,nsp*ngrid)
    rho[rho < 0.0] = 0.0 
    rho[:ngrid] *= totaldensity/np.sum(rho[:ngrid])
    rho[ngrid:2*ngrid] *= totaldensity/np.sum(rho[ngrid:2*ngrid])
    fe = totalfe(rho)
    feovertime.append(fe)
    p1 = np.exp(-fe)
    if p1 > p0:
#        print('accepted p1 = %.3e p0 = %.3e'%(p1,p0))
        return rho, p1
    else:
        if np.random.uniform(0.0,1.0) < p1/p0:
#            print('accepted p1/p0 = %.3e'%(p1/p0))
            return rho, p1
        else:
#            print('rejected p1 = %.3e p0 = %.3e'%(p1,p0))
            return x, p0

np.random.seed(0)
plt.plot(x,rho[:ngrid],color='r')
plt.plot(x,rho[ngrid:2*ngrid],color='b')
plt.show()
plt.close()
print(totalfe(rho))
p0 = np.exp(-totalfe(rho))
for i in range(mciter):
    rho, p0 = TakeStep(rho,p0)
print(totalfe(rho))
np.savetxt('rho.dat',rho)

data = np.loadtxt('rho.dat')
rho = data
plt.plot(x,rho[:ngrid],color='r')
plt.plot(x,rho[ngrid:2*ngrid],color='b')
plt.show()
plt.close()

plt.plot(range(len(feovertime)),feovertime)
plt.show()
plt.close()