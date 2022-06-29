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
    band = 0
    xi = 1
    chern_band = ['-1', '+2', '-2', '+1']
    
    fbz_mesh = make_kmesh(Rgrid)
    
    temp = 0. #K

    print('------ CALCULATING FOR BAND {0:} ------'.format(band))
    mulim.set_lims(band)
    
    mu_list = np.append(np.linspace(mulim.lims[0], band_top[band], int(Nmu/2)), np.linspace(band_top[band], mulim.lims[1], int(Nmu/2))) 
    savename = 'data/Mz_susc-band' + str(band) + '_R' + str(Rgrid)

    for point in fbz_mesh:
        k = kMvec(point, 'FBZ')
        H = Hxi(k, 0, xi)
        
        print('k = {0}'.format(k.coords))
        
        evals, evecs = eigsh(H, k=100, which='SM')
    
        del H
        
        Bc = Berry_curv(k, band, evals, evecs, xi)
            
        try:
            Bc_array = np.vstack((Bc_array, Bc))
        except:
            Bc_array = Bc
            
        try:
            evals_array = np.vstack((evals_array, evals))
        except:
            evals_array = evals
    
    np.savez(savename, Bc = Bc_array, evals = evals_array)
                  
    bcq = BerryCurvQuantities(Bc_array, band, mu_list, temp)
    
    chern = bcq.get_Bc(evals_array[:,band])
    Mz_int = bcq.get_Mz_int(evals_array[:,band])
    Mz_edge = bcq.get_Mz_edge(evals_array[:,band])
    susc = bcq.get_orb_susc(evals_array[:,band])
    
    title = 'Band {}'.format(band + 1) + ', C = ' + chern_band[band]
    
    plot_grid(fbz_mesh, Bc_array[:, 0], grid='FBZ', title = title, cvalue = 'Berrycurv')
    plot_grid(fbz_mesh, Bc_array[:, 1], grid='FBZ', title = title)
    plot_grid(fbz_mesh, Bc_array[:, 0]*Bc_array[:, 1], grid='FBZ', title = title)
    
    plot_vs_mu(mu_list, Mz_int, 'Mz', mu_list[Nmu // 2], title = 'Intrinsic magnetization\n' + title)
    plot_vs_mu(mu_list, Mz_edge, 'Mz', mu_list[Nmu // 2], title = 'Edge magnetization\n' + title)
    plot_vs_mu(mu_list, Mz_int + Mz_edge, 'Mz', mu_list[Nmu // 2], title = 'Total magnetization\n' + title)
    plot_vs_mu(mu_list, susc, 'susc', mu_list[Nmu // 2], title = 'Susceptibility\n' + title)
    plot_vs_mu(mu_list, chern, r'$\Omega$', mu_list[Nmu // 2], title = 'Berry curvature\n' + title)
    