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
import matplotlib

matplotlib.rcParams["mathtext.rm"] = 'serif'
matplotlib.rcParams["mathtext.fontset"] = 'cm'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Century'

# set the font globally
plt.rcParams.update({'font.family':'serif'})

if __name__ == '__main__':
    band_top = np.load('data/band_lims_R6.npz')['top']
    band_bottom = np.load('data/band_lims_R6.npz')['bottom']
    # Set chemical potential to one bandwidth above the top of the band    
    fbz_mesh = make_kmesh(Rgrid)
    spin = True
    
    for band in range(1,4):
        savename = 'Mz' + '_band' + str(band) + '_R' + str(Rgrid)
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
            
            if spin:
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

        
        if spin:
            Szavg = np.average(np.real(Sz_array))
            plot_grid(fbz_mesh, np.real(Sz_array), grid='FBZ', title=r'$\left\langle S_z \right\rangle = $' + str(Szavg), plottype = 'contour')
            print('Sz list shape is: {0:}'.format(Sz_array.shape))
            si_sz = SaveInfo(band, 'Sz')
            si_sz.save_data(Sz_array)
        
        bcq = BerryCurvQuantities(Bc_array, band, band_top[band])
        mk_int = Bc_array[:,1]
        mk_edge = bcq.get_mk_edge(evalslist[:,band])
        mk_tot = mk_int + mk_edge
        Mz = np.sum(mk_tot)/len(mk_tot)

        plot_grid(fbz_mesh, mk_tot, grid='FBZ', title=r'$M_z = ${0:.2f} $\mu_B$ / u.c.'.format(Mz), plottype = 'contour')
        
        plot_grid(fbz_mesh, mk_int, grid='FBZ', title=r'$M_z = ${0:.2f} $\mu_B$ / u.c.'.format(np.average(mk_int)), plottype = 'contour')
        plot_grid(fbz_mesh, mk_edge, grid='FBZ', title=r'$M_z = ${0:.2f} $\mu_B$ / u.c.'.format(np.average(mk_edge)), plottype = 'contour')
        
        print('Bc list shape is: {0:}'.format(Bc_array.shape))
        si = SaveInfo(band, 'Bc')
        si.save_data(Bc_array)

        print('Eigenlist shape is: {0:}'.format(evalslist.shape))
        si_evals = SaveInfo(band, 'evals')
        si_evals.save_data(evalslist)
        
        #plot_grid(kmesh, np.sum(Fxy_list, axis=0), grid='FBZ', title=r'$M_z = $' + str(Mz) + ' A $\AA^2$', savename = savename)
    