import matplotlib.pyplot as plt
import pyccl as ccl
import numpy as np
import astropy.table
import pandas as pd
import math
import pickle
from astroML.datasets import fetch_sdss_specgals
from scipy.optimize import minimize
import SDSS_treecorr as stc
import SDSS_treecorr2 as stc2



def getdata():
    data = fetch_sdss_specgals()
    m_max = 17.7
    # redshift and magnitude cuts
    data = data[data['z'] > 0.08]
    data = data[data['z'] < 0.12]
    data = data[data['petroMag_r'] < m_max]
    # RA/DEC cuts
    RAmin, RAmax = 140, 220 
    DECmin, DECmax = 5, 45
    data = data[data['ra'] < RAmax] 
    data = data[data['ra'] > RAmin] 
    data = data[data['dec'] < DECmax] 
    data = data[data['dec'] > DECmin]
    ur = data['modelMag_u'] - data['modelMag_r'] 
    flag_red = (ur > 2.22)
    flag_blue = ~flag_red
    data_red = data[flag_red] 
    data_blue = data[flag_blue]
    
    return [data, data_red, data_blue]



def getcorrTree(data):
    corr, bincenters, cov = stc.calccorr(data)
    return [corr, math.e**bincenters, cov]

def getcorrTree2(data, minsep, maxsep):
    corr, bincenters, cov = stc2.calccorr(data, minsep, maxsep)
    return [corr, math.e**bincenters, cov]



def getcorrCCL(theta, data, centers):
    
    Nz, be = np.histogram(data['z'], bins=8, range=(0.05,0.15))
    z = 0.5*(be[1:] + be[:-1])
   
    h=0.675 # Planck value for h (Hubble parameter)
    Ob = 0.044 # Planck value for Omega_b (Baryon energy density)
    Om = theta[1] # Planck value for Omega_m (Matter energy density)
    Oc = Om-Ob # Value for Omega_c (Cold dark matter energy density)
    ns=0.965 # Scalar index
    
    cosmo = ccl.Cosmology(Omega_c=Oc, Omega_b=Ob, h=h, sigma8=0.8, n_s=ns, matter_power_spectrum='linear')
    
    tracer = ccl.NumberCountsTracer(cosmo,has_rsd=False,
                                                dndz=(z,Nz),
                                                bias=(z,np.ones_like(z)))

    ell = np.arange(1, 7500) # is this the same as lmax?
    angular_power_spectrum = ccl.angular_cl(cosmo, tracer, tracer, ell)
    
    th = centers #np.linspace(0,0.2, num = 15)

    ang_corr_func = ccl.correlation(cosmo, ell, angular_power_spectrum, th)
    
    return ang_corr_func



def biasfunc2(theta, corr_tree, data, centers, inv_cov):
    corr_ccl = getcorrCCL(theta, data, centers)
    return np.einsum('i,i',(corr_tree- theta[0]**2*corr_ccl),
                     np.einsum('ij,j', inv_cov,(corr_tree- theta[0]**2*corr_ccl)))

"""
def biasfunc2(b, corr_tree, corr_camb, inv_cov):
    return np.einsum('i,i',(corr_tree- b**2*corr_camb), np.einsum('ij,j', inv_cov,(corr_tree- b**2*corr_camb)))

"""


def findb(data):
    corr, centers, cov = getcorrTree(data)
    corr_camb = getcorrCCL([1, 0.31], data, centers)
    
    corr_tree = corr[0]
    
    inv_cov = np.linalg.inv(cov)
    
    result = minimize(biasfunc2, np.array([0.1,0.1]), args = (corr_tree, data, centers, inv_cov), bounds = ((0.6,4),(0,1)));
    
    return result

def findb2(data, minsep, maxsep):
    corr, centers, cov = getcorrTree2(data, minsep, maxsep)
    corr_camb = getcorrCCL([1, 0.31], data, centers)
    
    corr_tree = corr[0]
    
    inv_cov = np.linalg.inv(cov)
    
    result = minimize(biasfunc2, np.array([1,0.31]), args = (corr_tree, data, centers, inv_cov), bounds = ((0.6,4),(0.1,1)));
    
    return result