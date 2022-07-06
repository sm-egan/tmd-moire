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
    band = 0
    xi = 1
    chern_band = ['-1', '+2', '-2', '+1']
    spin = True
    
    fbz_mesh = make_kmesh(Rgrid)
    
    print('------ CALCULATING FOR BAND {0:} ------'.format(band))
    savename = 'data/Mk_data-band' + str(band) + '_R' + str(Rgrid)

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
            
        if spin:
            Sz = exp_spin(evals, evecs, band)
            #print('Sz for point ' + str(point) + ' is: ' + str(Sz))
            try:
                Sz_array = np.append(Sz_array, Sz)
            except:
                Sz_array = np.array([Sz])
    
    if spin:
        np.savez(savename, Bc = Bc_array, evals = evals_array, Sz = Sz_array)
    else:
        np.savez(savename, Bc = Bc_array, evals = evals_array)
                  
    bcq = BerryCurvQuantities(Bc_array, band, mu_list, temp)
    
    title = 'Band {}'.format(band + 1) + ', C = ' + chern_band[band]
    
    plot_grid(fbz_mesh, Bc_array[:, 0], grid='FBZ', title = title, cvalue = 'Berrycurv')
    plot_grid(fbz_mesh, Bc_array[:, 1], grid='FBZ', title = title)
    plot_grid(fbz_mesh, Bc_array[:, 0]*Bc_array[:, 1], grid='FBZ', title = title)