# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 14:57:05 2021

@author: shann
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from tTMDpresets import *
import matplotlib.pyplot as plt  
plt.rcParams["figure.figsize"] = (9,8)
import time

if __name__ == '__main__':
    lims_file = 'data/band_lims_R6.npz'
    band_top = np.load('data/band_lims_R6.npz')['top']
    murange = 'bandplusgap'
    
    mulim = muLimits(lims_file, 0, murange)
    Nmu = 50 # How many mus to scan
    band = 1
    
    fbz_mesh = make_kmesh(Rgrid)
    
    temp = 0. #K

    print('------ CALCULATING FOR BAND {0:} ------'.format(band))
    mulim.set_lims(band)
    
    mu_list = np.append(np.linspace(mulim.lims[0], band_top[band], int(Nmu/2)), np.linspace(band_top[band], mulim.lims[1], int(Nmu/2))) 
    savename = 'data/Mz_susc_mu-band' + str(band) + '_R' + str(Rgrid) + '_T{0:.0e}'.format(temp).replace('.','_').replace('+', '')

    for point in fbz_mesh:
        k = kMvec(point, 'FBZ')
        H = Hxi(k, 0, -1)
        
        print('k = {0}'.format(k.coords))
        
        evals, evecs = eigsh(H, k=100, which='SM')
    
        del H
        
        Bc = Berry_curv(k, band, evals, evecs)
            
        try:
            Bc_array = np.vstack((Bc_array, Bc))
        except:
            Bc_array = Bc
            
        try:
            evals_array = np.vstack((evals_array, evals))
        except:
            evals_array = evals
    
    Mz_list = []
    susc_list = []
                  
    bcq = BerryCurvQuantities(Bc_array, band, mu_list, temp)
        
    Mz_int = bcq.get_Mz_int(evals_array[:,1])
    Mz_edge = bcq.get_Mz_edge(evals_array[:,1])
    susc = bcq.get_orb_susc(evals_array[:,1])
        
    #np.savez(savename, Mz = Bc_array, mu = mu_list, evals = evals_array)
    
    plot_grid(fbz_mesh, Bc_array[:, 0], grid='FBZ')
    plot_grid(fbz_mesh, Bc_array[:, 1], grid='FBZ')
    plot_grid(fbz_mesh, Bc_array[:, 0]*Bc_array[:, 1], grid='FBZ')
    
    plot_vs_mu(mu_list, Mz_int, 'Mz', mu_list[Nmu // 2])
    plot_vs_mu(mu_list, Mz_edge, 'Mz', mu_list[Nmu // 2])
    plot_vs_mu(mu_list, susc, 'susc', mu_list[Nmu // 2])
        