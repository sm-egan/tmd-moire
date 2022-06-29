# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 16:28:06 2022

@author: shann
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from tTMDpresets import *
import matplotlib.pyplot as plt  
import matplotlib.cm as cm
plt.rcParams["figure.figsize"] = (9,8)
import time

if __name__ == '__main__':
    lims_file = 'data/band_lims_R6.npz'
    band_top = np.load('data/band_lims_R6.npz')['top']
    band_bottom = np.load('data/band_lims_R6.npz')['bottom']
    bandwidth = band_top - band_bottom
    chern_band = ['-1', '+2', '-2', '+1']
    
    fbz_mesh = make_kmesh(Rgrid)
    
    Nmu = 1000
    mu_list = np.linspace(-26.5, -18.5, Nmu)
    
    temp = np.array([0, 0.2, 0.5, 0.7, 1, 1.2]) #K
    
    susc_T = np.zeros((len(temp), 4, Nmu))
    Mz_int_T = np.zeros((len(temp), 4, Nmu))
    Mz_edge_T = np.zeros((len(temp), 4, Nmu))
    
    i = 0
    for T in temp:
        for band in range(0,4):
            savename = 'data/Mz_susc_mu-band' + str(band) + '_R' + str(Rgrid)
            data = np.load(savename + '.npz')
            Bc_array = data['Bc']
            evals_array = data['evals']
            
            bcq = BerryCurvQuantities(Bc_array, band, mu_list, T)
            Bc = bcq.get_Bc(evals_array[:,band])
            Mz_int = bcq.get_Mz_int(evals_array[:,band])
            Mz_edge = bcq.get_Mz_edge(evals_array[:,band])
            susc = bcq.get_orb_susc(evals_array[:,band], True)
            if band == 0:
                mu_mask = (mu_list > band_bottom[band] - bandwidth[band]) & (mu_list < band_top[band] + 2*bandwidth[band])
            else:
                mu_mask = (mu_list > band_bottom[band] - bandwidth[band]) & (mu_list < band_top[band] + bandwidth[band])
            title = 'Band {}'.format(band + 1) + ', C = ' + chern_band[band] + ', T = ' + str(T)
            #plot_vs_mu(mu_list[mu_mask], Auc_nm*(Mz_int + Mz_edge)[mu_mask], band_region = [band_bottom[band], band_top[band]], title = title, ytype = 'Mz_muB')
            # if T == 0.2:
            #     plot_Mz_vs_mu(mu_list[mu_mask], Auc_nm*Mz_int[mu_mask], Auc_nm*Mz_edge[mu_mask], band_region = [band_bottom[band], band_top[band]], title = title, ytype = 'Mz_muB')
            #     title = 'Band {}'.format(band + 1) + ', C = ' + chern_band[band]
            #     plot_grid(fbz_mesh, AAng_to_muB*Bc_array[:,0], grid='FBZ', title = title, cvalue='Berrycurv')
            #     plot_grid(fbz_mesh, AAng_to_muB*Bc_array[:,1], grid='FBZ', title = title)
                
            if band == 0:
                Bc_arr = Bc
                Mz_int_arr = Mz_int
                Mz_edge_arr = Mz_edge
                susc_arr = susc
            else:
                Bc_arr = np.vstack((Bc_arr, Bc))
                Mz_int_arr = np.vstack((Mz_int_arr, Mz_int))
                Mz_edge_arr = np.vstack((Mz_edge_arr, Mz_edge))
                susc_arr = np.vstack((susc_arr, susc))
            
        susc_T[i] = susc_arr
        Mz_int_T[i] = Mz_int_arr
        Mz_edge_T[i] = Mz_edge_arr
        
        i += 1
            
    Mz_total = Mz_int_arr + Mz_edge_arr
    # Mz_int_sum = np.sum(Mz_int_arr, axis = 0)
    # Mz_edge_sum = np.sum(Mz_edge_arr, axis = 0)
    # Mz_total_sum = np.sum(Mz_total, axis = 0)
    susc_sum = np.sum(susc_T[:,0:4], axis = 1)
           
    Bc = 1/(3*np.sqrt(3))*np.abs(Mz_total[1])/susc_sum*eV/muB
    Bc_mask = (susc_sum > 0) & (Bc < 20)
    
    band = 1
    title = 'Band {}'.format(band) + ', C = ' + chern_band[band] + ', T = ' + str(temp[2])
    plot_vs_mu(mu_list, susc_T[2,band], 'susc', band_region = [band_bottom[band], band_top[band]], title = title, xlims = [-25.5, -20], ylims = [-0.5, 3])

    #############################################
    ########## Plot Bc at multiple T ############
    Tind = [1, 3, 4, 5]
    for i in range(0, len(Tind)):
        ind = Tind[i]
        plt.plot(mu_list[Bc_mask[ind]], Bc[ind][Bc_mask[ind]], 'o', color = cm.gnuplot2((i+1)/(len(Tind)+1)), label = 'T = ' + str(temp[ind]), markersize=8)
    plt.yscale('log')
    plt.legend(fontsize = 20, loc = 2)
    plt.axvspan(band_bottom[1], band_top[1], color='grey', alpha=0.4)
        
    plt.xlabel(r'$\mu$ (meV)', size='xx-large')
    plt.ylabel(r'$B_c$ (T)', size = 'xx-large')
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlim(-24.5, -22)
    plt.ylim(1e-1, 20)
    plt.grid(axis='both', which='both')
    plt.show()
    
    #############################################
    ########## Plot susc at multiple T ############
    Tind2 = [1, 3, 4, 5]
    for i in range(0, len(Tind2)):
        ind = Tind2[i]
        plt.plot(mu_list, susc_sum[ind], linewidth = 3, color = cm.gnuplot2((i+1)/(len(Tind2)+1)), label = 'T = ' + str(temp[ind]))
    plt.legend(fontsize = 20, loc = 1)
    plt.axvspan(band_bottom[1], band_top[1], color='grey', alpha=0.4)
    plt.axhline(0, c='k', linestyle='--')
        
    plt.xlabel(r'$\mu$ (meV)', size='xx-large')
    plt.ylabel(r'$\chi_{orb}$ ($\mu_B \, eV^{-1} \, nm^{-2}$)', size = 'xx-large')    
    
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    #plt.xlim(-24.5, -22)
    plt.grid(axis='both', which='both')
    plt.show()

    
    
