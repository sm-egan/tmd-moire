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
    Nmu = 24 # How many mus to scan
    band = 1
    
    fbz_mesh = make_kmesh(Rgrid)
    
    temp = 0. #K

    print('------ CALCULATING FOR BAND {0:} ------'.format(band))
    mulim.set_lims(band)
    
    mu_list = np.append(np.linspace(mulim.lims[0], band_top[band], int(Nmu/2)), np.linspace(band_top[band], mulim.lims[1], int(Nmu/2))) 
    savename = 'data/Mz_susc_mu-band' + str(band) + '_R' + str(Rgrid) + '_T{0:.0e}'.format(temp).replace('.','_').replace('+', '')

    for point in fbz_mesh:
        k = kMvec(point, 'FBZ')
        Hplus = Hxi(k, 0)
        
        print('k = {0}'.format(k.coords))
        
        evals, evecs = eigsh(Hplus, k=100, which='SM')
    
        del Hplus
        
        for mu in mu_list:
            Bc = Berry_curv(k, mu, band, evals, evecs)
            
            try:
                Bc_array = np.vstack((Bc_array, Bc))
            except:
                Bc_array = Bc
                
            try:
                evals_array = np.vstack((evals_array, evals))
            except:
                evals_array = evals
            
            #print(Bc_array)
            #print(evals_array)
                
    # Reshape the data and swap axes of k, mu dof
    ### We do this so that Bc_mu_array[i] is all the data at fixed mu for different k values
    Bc_mu_array = np.swapaxes(Bc_array.reshape((len(fbz_mesh), Nmu, 3)), 0, 1)
    evals_mu_array = np.swapaxes(evals_array.reshape((len(fbz_mesh), Nmu, 100)), 0, 1)
    
    Mz_list = []
    susc_list = []
    
    for i in range(0, Nmu):               
        bcq = BerryCurvQuantities(Bc_mu_array[i], band, mu_list[i], temp)
        
        Mz_list.append(bcq.get_Mz(evals_mu_array[i]))
        susc_list.append(bcq.get_orb_susc(evals_mu_array[i]))
        
    #Mzs = np.sum(Fxy_list, axis=0)/len(fbz_mesh)
    np.savez(savename, Mz = Bc_mu_array, susc = susc_list, mu = mu_list)
    
    plot_grid(fbz_mesh, np.sum(Bc_mu_array[15][:,1:3], axis=1)*AAng_to_muB, grid='FBZ')
    plot_grid(fbz_mesh, Bc_mu_array[int(Nmu/2)][:, 0], grid='FBZ')
    plot_grid(fbz_mesh, Bc_mu_array[int(Nmu/2)][:, 1], grid='FBZ')
    plot_grid(fbz_mesh, Bc_mu_array[int(Nmu/2)][:, 0]*Bc_mu_array[15][:, 1], grid='FBZ')
    
    plot_vs_mu(mu_list, np.array(Mz_list), 'Mz', mu_list[int(Nmu/2)])
    plot_vs_mu(mu_list, np.array(susc_list), 'susc', mu_list[int(Nmu/2)])
        