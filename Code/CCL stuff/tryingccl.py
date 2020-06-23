import pyccl as ccl
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import astropy.table
import pickle

data = astropy.table.Table.read('/home/s1/kkrzyzan/LSST_i.fits')

cut1 = pd.cut(data['z'], bins = 25)
Nz = cut1.value_counts().tolist() #z=np.linspace(0,1)
#Nz = np.exp(-0.5*((z-0.5)**2/(0.1**2)))

keys1 = cut1.value_counts().keys()


z = []
for i in range(len(keys1)):
    z.append(keys1[i].mid)


with open('z1.data', 'wb') as filehandle:
    pickle.dump(z, filehandle)

with open('Nz1.data','wb') as filehandle:
    pickle.dump(Nz,filehandle)

cosmo = ccl.Cosmology(Omega_c=0.27, Omega_b=0.044, h=0.7, sigma8=0.8, n_s=0.96)


plt.scatter(z, Nz)
plt.show()


tracer = ccl.NumberCountsTracer(cosmo,has_rsd=False,
                                                dndz=(z,Nz),
                                                bias=(z,np.ones_like(z)))

ell = np.arange(1, 20000)
angular_power_spectrum = ccl.angular_cl(cosmo, tracer, tracer, ell)

"""
In astropy, theta was bincenters = nn.meanlogr, 
taking the mean value of log(r) for the pair in the bin. 
I'm not sure how to best define pairs per bin- would it literally
be finding the distance from every galaxy to every other galaxy?-
so I'm going to make a dummy set of theta values.

"""

#theta = np.logspace(10**(-2), 10**(-1), num = 10)

theta = np.linspace(0,0.2, num = 15)

ang_corr_func = ccl.correlation(cosmo, ell, angular_power_spectrum, theta)

with open('ccl_theta1', 'wb') as filehandle:
    pickle.dump(theta, filehandle)

with open('ccl_corr1', 'wb') as filehandle:
    pickle.dump(ang_corr_func, filehandle)

plt.scatter(theta, ang_corr_func)

plt.show()
