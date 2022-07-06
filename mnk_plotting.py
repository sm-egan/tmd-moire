# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 16:08:39 2022

@author: shann
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from tTMDpresets import *
import matplotlib.pyplot as plt  
import matplotlib.cm as cm
import matplotlib

matplotlib.rcParams["mathtext.rm"] = 'serif'
matplotlib.rcParams["mathtext.fontset"] = 'cm'
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = 'Century'

# set the font globally
plt.rcParams.update({'font.family':'serif'})

if __name__ == '__main__':
    lims_file = 'data/band_lims_R6.npz'
    band_top = np.load('data/band_lims_R6.npz')['top']
    band_bottom = np.load('data/band_lims_R6.npz')['bottom']
    bandwidth = band_top - band_bottom
    chern_band = ['-1', '+2', '-2', '+1']
    band = 2
    
    fbz_mesh = make_kmesh(Rgrid)
    
    savename = 'data/Bc_band' + str(band) + '_R' + str(Rgrid)
    data = np.load(savename + '.npz')
    Bc_array = data['Bc']
    evals_array = data['evals']
    T = 0
    
    bcq = BerryCurvQuantities(Bc_array, band, band_top[band])
    Omega = Bc_array[:, 0]
    mk_int = Bc_array[:,1]*Auc_nm
    mk_edge = bcq.get_mk_edge(evals_array[:,band])
    mk_tot = mk_int + mk_edge
    
    ###########################################################
    #### Plot orbital magnetic moments as contour plot ########
    xlist = []
    ylist = []
    for row in fbz_mesh:
        xlist.append(kMvec(row, FBZgrid).get_x())
        ylist.append(kMvec(row, FBZgrid).get_y())
    
    intmax = np.max(np.abs(mk_int))
    edgemax = np.max(np.abs(mk_edge))
    summax = np.max(np.abs(mk_tot))
    
    ticks = np.array([[0, 0.54, 25, 10],[0, 3.5, 21, 8],[-8.4, 0, 19, 8],[-0.2, 2.4, 21, 13]])
    manualticks = True
    
    fig1, ax1 = plt.subplots()
    if manualticks:
        tcf = ax1.tricontourf(xlist, ylist, mk_tot, levels = np.linspace(ticks[band,0], ticks[band,1], int(ticks[band,2])), cmap='Spectral', vmin=-summax, vmax=summax)
        if band == 3:
            cbar = fig1.colorbar(tcf, ticks = [-0.2, 0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4], pad = 0.1)
        else:
            cbar = fig1.colorbar(tcf, ticks = np.linspace(ticks[band,0], ticks[band,1], int(ticks[band,3])), pad = 0.1)
    else:
        tcf = ax1.tricontourf(xlist, ylist, mk_tot, levels = 20, cmap='Spectral', vmin = -summax, vmax = summax)
        cbar = fig1.colorbar(tcf)
        
    ax1.set_aspect(1)
    # ax1.set_xticks([])
    # ax1.set_yticks([])
    ax1.axis('off')
    ax1.set_title(r'$M_z =$ {0:.2f} $\mu_B$ / u.c.'.format(np.average(mk_tot)), size = 25, pad = 10)
    cbar.ax.tick_params(labelsize = 16)
    cbar.set_label(r'$m_{n\mathbf{k}}^{tot} \, (\mu_B)$', size = 29)
    fig1.savefig('plots/mk_tot-band{}.svg'.format(band), transparent = True, bbox_inches='tight', pad_inches = 0.2)