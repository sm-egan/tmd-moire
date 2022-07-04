# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 14:52:18 2021

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
    band_top = np.load('data/band_lims_R6.npz')['top']
    band_bottom = np.load('data/band_lims_R6.npz')['bottom']
    # Set chemical potential to one bandwidth above the top of the band
    mus = 2*band_top - band_bottom
    
    fbz_mesh = make_kmesh(Rgrid)
    
    for band in range(1,2):
        mu = mus[band]
        
        savename = 'Mz' + '_band' + str(band) + '_R' + str(Rgrid) + '_mu{:+.2f}'.format(mu)
        savename = savename.replace('.', '_')
        print('BAND = ' + str(band))
        
        
        for point in fbz_mesh:
            k = kMvec(point, 'FBZ')
            # The second argument is the chemical potential.  We set it to zero here so that we can have a common 
            # reference point for the band energy as we modify mu later 
            Hplus = Hxi(k, 0)
            
            evals, evecs = eigsh(Hplus, k=100, which='SM') # 'SM' means take smallest magnitude eigenvalues
            #print('Eigenvalue problem time: ' + str(time.process_time() - start))             
            
            Bc = Berry_curv(k, band, evals, evecs)
            try:
                Bc_array = np.vstack((Bc_array, Bc))
            except:
                Bc_array = Bc
            
            Sz = exp_spin(evals, evecs, band)
            #print('Sz for point ' + str(point) + ' is: ' + str(Sz))
            try:
                Sz_array = np.append(Sz_array, Sz)
            except:
                Sz_array = np.array([Sz])
            
            try:
                evalslist = np.vstack((evalslist, evals))
            except:
                evalslist = evals                    
            
            #print(eigenlist.shape)
            del Hplus, evals, evecs

        Mz = AAng_to_muB*np.sum(Bc_array[:,1:3])/(len(Bc_array)*Auc_nm)
        Szavg = np.mean(Sz_array).real

        plot_grid(fbz_mesh, np.sum(Bc_array[:,1:3], axis=1)*AAng_to_muB, grid='FBZ', title=r'$M_z = $' + str(Mz) + ' $\mu_B/nm^2$', savename = savename)
        plot_grid(fbz_mesh, Sz_array, grid='FBZ', title=r'$\left\langle S_z \right\rangle = $' + str(Szavg))
        
        print('Bc list shape is: {0:}'.format(Bc_array.shape))
        si = SaveInfo(band, 'Bc_mu{0:3d}'.format(int(mu*10)))
        si.save_data(Bc_array)
        
        print('Sz list shape is: {0:}'.format(Sz_array.shape))
        si_sz = SaveInfo(band, 'Sz')
        si_sz.save_data(Sz_array)
        
        print('Eigenlist shape is: {0:}'.format(evalslist.shape))
        si_evals = SaveInfo(band, 'evals')
        si_evals.save_data(evalslist)
        
        #plot_grid(kmesh, np.sum(Fxy_list, axis=0), grid='FBZ', title=r'$M_z = $' + str(Mz) + ' A $\AA^2$', savename = savename)
    