import treecorr

from random import *
import math
import numpy as np
import astropy.table
from matplotlib import pyplot as plt
from astroML.datasets import fetch_sdss_specgals

# Generates random data to create correlation function
def randomcorr(dataset, cat):
    
    ra_range = max(cat.ra) - min(cat.ra)
    dec_range = max(cat.dec) - min(cat.dec)
    ra_min = min(cat.ra)
    dec_min = min(cat.dec)
    
    rand_ra = []
    rand_dec = []

    min_cth = np.min(np.sin(cat.dec))
    max_cth = np.max(np.sin(cat.dec))
    rand_dec = np.arcsin(min_cth+(max_cth-min_cth)*np.random.random(size=len(dataset))) 

    for i in range(len(dataset)):
        u = random()
        rand_ra.append((2*math.pi*u)*(ra_range/(2*math.pi)) + ra_min)
        
    rr = treecorr.NNCorrelation(nbins = 16, min_sep = 1/60, max_sep = 6, sep_units = 'deg')
    catrand = treecorr.Catalog(ra = rand_ra, dec = rand_dec, ra_units='rad', dec_units='rad')
    
    return catrand;

# Creates plot of correlation function
def plotcorr(dataset1, dataset2, bin_centers, corr, corr_stdev, error):
    labels = ['$u-r > 2.22$\n$N=%i$' % len(dataset1),
          '$u-r < 2.22$\n$N=%i$' % len(dataset2)]
    
    fig = plt.figure(figsize=(5, 2.5))
    fig.subplots_adjust(bottom=0.2, top=0.9,
                        left=0.13, right=0.95)

    for i in range(2):
        ax = fig.add_subplot(121 + i, xscale='log', yscale='log')

        ax.set_ylim([10**(-2),10])
        ax.set_yscale('log')

        ax.set_xlim([10**(-2),10])
        ax.set_xscale('log')
    

        if error:
            ax.errorbar(math.e**bin_centers[i], corr[i][0], corr_stdev[i], fmt='.k', ecolor='gray', lw=1)
        else:
            plt.scatter(math.e**(bin_centers[i]), corr[i][0])
        
        t = np.array([0.01, 10])
        ax.plot(t, 10 * (t / 0.01) ** -0.8, ':k', linewidth=1)

        ax.text(0.95, 0.95, labels[i],
                ha='right', va='top', transform=ax.transAxes)
        ax.set_xlabel(r'$\theta\ (deg)$')
        if i == 0:
            ax.set_ylabel(r'$\hat{w}(\theta)$')


# Calculates the correlation function needed for plotting
def calccorr(dataset):
    
    nn = treecorr.NNCorrelation(nbins = 16, min_sep = 1/60, max_sep = 6, sep_units = 'deg',var_method='jackknife') 
    cat = treecorr.Catalog(ra = dataset['ra'], dec = dataset['dec'], ra_units='deg', dec_units='deg',npatch = 100)
    nn.process(cat)

    catrand = randomcorr(dataset, cat)
    rr = treecorr.NNCorrelation(nbins = 16, min_sep = 1/60, max_sep = 6, sep_units = 'deg',var_method='jackknife')
    rr.process(catrand)
    
    dr = treecorr.NNCorrelation(nbins = 16, min_sep = 1/60, max_sep = 6,sep_units = 'deg',var_method='jackknife')
    dr.process(cat, catrand)

    corr = nn.calculateXi(rr, dr)
    bin_centers = nn.meanlogr
    cov = nn.estimate_cov('jackknife')
    
    return [corr, bin_centers, cov]


# Taking the file path of the data, thte requested purity, and the bounds of the
# redshift (red1 and red2), this produces the plot.
def producecorrplots(error = False):
    data = getdata()[0]
    data_red = getdata()[1]
    data_blue = getdata()[2]

    calc_red = calccorr(data_red)
    calc_blue = calccorr(data_blue)

    corr = [calc_red[0], calc_blue[0]]
    bin_centers = [calc_red[1], calc_blue[1]]

    corr_stdev_red = [math.sqrt(k) for k in corr[0][1]]
    corr_stdev_blue = [math.sqrt(k) for k in corr[1][1]]
    corr_stdev = [corr_stdev_red, corr_stdev_blue]
    
    plotcorr(data_red, data_blue, bin_centers, corr, corr_stdev, error)

# Getting the data and cleaning it up a bit
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

