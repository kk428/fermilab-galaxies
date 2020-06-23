import matplotlib.pyplot as plt
import camb
from camb.sources import GaussianSourceWindow, SplinedSourceWindow
import camb.correlations
from camb import model, initialpower
import numpy as np
import astropy.table
import pandas as pd
import math
import pickle
from astroML.datasets import fetch_sdss_specgals
from scipy.optimize import fmin
import SDSS_treecorr as stc


# Cleans up SDSS data a bit
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


# Finds corr. function with tree corr
def getcorrTree(data):
    corr, bincenters, cov = stc.calccorr(data)
    return [corr, math.e**bincenters, cov]


# Finds corr. function with CAMB
def getcorrCAMB(theta, data, centers):
    Nz, be = np.histogram(data['z'], bins=8, range=(0.05,0.15))
    lmax = 3750
    pars = camb.CAMBparams() # Set up the CAMB parameters
    h=0.675 # Planck value for h (Hubble parameter)
    Ob = 0.044 # Planck value for Omega_b (Baryon energy density)
    Om = theta[1] # Planck value for Omega_m (Matter energy density)
    Oc = Om-Ob # Value for Omega_c (Cold dark matter energy density)
    As=2e-9 # Amplitude of initial fluctuations
    ns=0.965 # Scalar index
    pars.set_cosmology(H0=100*h, ombh2=Ob*h**2, omch2=Oc*h**2) # This sets the cosmological parameters
    pars.InitPower.set_params(As=As, ns=ns) # This also sets the cosmological parameters
    pars.set_for_lmax(lmax, lens_potential_accuracy=1) # Set the maximum ell
    #set Want_CMB to true if you also want CMB spectra or correlations
    pars.Want_CMB = False # We don't want the CMB
    #NonLinear_both or NonLinear_lens will use non-linear corrections
    pars.NonLinear = model.NonLinear_both # We want non-linear corrections
    #Set up W(z) window functions, later labelled W1, W2.
    zs = 0.5*(be[1:] + be[:-1]) #z # Range of zs
    W = Nz # Window function
    pars.SourceWindows = [SplinedSourceWindow(source_type='counts', bias=theta[0], z=zs, W=W)] # Set up the window function
    
    results = camb.get_results(pars)
    cls = results.get_source_cls_dict()
    ls=  np.arange(2, lmax+1)
    
    angles = centers #np.logspace(-2, 1) # Angles from 0.01 to 10 deg
    x = np.cos(np.radians(angles)) # Convert them to radians and compute cosine to passs to CAMB
    cls_in = np.array([cls['W1xW1'][1:lmax+1], np.zeros(lmax), np.zeros(lmax), np.zeros(lmax)]).T
    #cl2corr needs TT (temperature/density), EE (E-mode), BB (B-mode), TE (Temperature-polarization cross correlation) -> we only care about TT
    w_camb = camb.correlations.cl2corr(cls_in, x);
    
    return w_camb


# Used to find minimum bias
def biasfunc2(b, corr_tree, corr_camb, inv_cov):
    return np.einsum('i,i',(corr_tree- b**2*corr_camb), np.einsum('ij,j', inv_cov,(corr_tree- b**2*corr_camb)))


# Finds minimum bias
def findb(data):
    corr, centers, cov = getcorrTree(data)
    w_camb = getcorrCAMB([1, 0.31], data, centers)
    
    corr_tree = corr[0]
    corr_camb = w_camb[:,0]
    
    inv_cov = np.linalg.inv(cov)
    
    result = fmin(biasfunc2, 0, args = (corr_tree, corr_camb, inv_cov), disp=False);
    
    return result[0]
    


