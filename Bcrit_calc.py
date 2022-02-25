# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 13:31:39 2021

@author: shann
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from tTMDpresets import *
import matplotlib.pyplot as plt  
plt.rcParams["figure.figsize"] = (9,8)
#from tabulate import tabulate

if __name__ == '__main__':
    band = 1
    temp = 1 # 1e-3 = 1 mK
    
    Bc_array = np.load('data/Bc_mu-211_band1_R20.npz')['data']
    evals = np.load('data/evals_band{0:}_R20.npz'.format(band))['data']
    band_top = np.load('data/band_lims_R6.npz')['top']
    band_bottom = np.load('data/band_lims_R6.npz')['bottom']    
        
    mu = 2*band_top - band_bottom
    
    fbz_mesh = make_kmesh(20)
    
    #orb_integrand = (2*e*eV)/hbar*Bc_array[:,0]*np.sum(Bc_array[:,1:2], axis=1)*AAng_to_muB**2
    orb_integrand = (2*e*eV)/hbar*Bc_array[:,0]*Bc_array[:,1]*AAng_to_muB**2
    
    plot_grid(fbz_mesh, Bc_array[:,0]*1e-2, grid='FBZ')
    plot_grid(fbz_mesh, Bc_array[:,1]*AAng_to_muB, grid='FBZ')
    plot_grid(fbz_mesh, orb_integrand, grid='FBZ')
    
    #plot_grid(fbz_mesh, np.sum(Bc_array[:,1:2], axis=1)*AAng_to_muB, grid='FBZ')
    # plot_grid(fbz_mesh, Bc_array[:,2], grid='FBZ')
    # plot_grid(fbz_mesh, Bc_array[:,1] + Bc_array[:,2], grid='FBZ')
    
    bcq = BerryCurvQuantities(Bc_array, band, mu[band], temp)
    print('Mz = {0:5f}, chi = {1:5f}'.format(bcq.get_Mz(evals), bcq.get_orb_susc(evals, 'int')))
    
    glfe = GLFreeEnergy(Bc_array, evals, band, mu[band], temp, 'int')
    glarray = glfe.param_array()
    print(glarray)
    