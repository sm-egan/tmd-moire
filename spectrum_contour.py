# -*- coding: utf-8 -*-
"""
Created on Thu May 13 12:05:01 2021

@author: shann
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from tTMDpresets import *
import matplotlib.pyplot as plt  
plt.rcParams["figure.figsize"] = (8,8)
import time

if __name__ == '__main__':
    start = time.process_time()
    path, xticks, xticklabels = kpath('KGammaKKK', 0.5)
    print('Path length = ' + str(len(path)))
    
    eigenlistp = None
    #eigenlistm = None
    for point in path:
        k = kMvec(point, 'FBZ')
        #print('kx = ' + str(k.get_x()))
        #print('ky = ' + str(k.get_y()))
        
#        H0p = Hd(k, 1)
#        HTp = HTeff(1)
#        Vmat = VM()
#        
#        H0m = Hd(k,-1)
#        HTm = HTeff(-1)
#        
#        Hplus = H0p + HTp + Vmat
#        Hminus = H0m + HTm + Vmat
        Hplus = Hxi(k, 0, 1)
        Hminus = Hxi(k,-1)
        
        evalsp, evecsp = eigsh(Hplus, k=24, which='SM')
        evalsp = np.sort(evalsp)
        
        evalsm, evecsm = eigsh(Hminus, k=24, which='SM')
        evalsm = np.sort(evalsm)
        
        try:
            eigenlistp = np.vstack((eigenlistp, evalsp))
            eigenlistm = np.vstack((eigenlistm, evalsm))
        except:
            eigenlistp = evalsp
            eigenlistm = evalsm
    
    band_top = np.array([])
    band_bottom = np.array([])
    
    for i in range(0,4):
        band_top = np.append(band_top, np.max(eigenlistp[:,i]))
        band_bottom = np.append(band_bottom, np.min(eigenlistp[:,i]))
    
    plot_label = 'R' + str(R) + '_Nk' + str(len(path))      
    plot_eigenlist(eigenlistp, xticks, xticklabels, 'o')
    plot_eigenlist(eigenlistm, xticks, xticklabels, 'o')
        
    np.savez('data/band_lims_R' + str(R), top = band_top, bottom = band_bottom)
        
    #plot_eigenlist(eigenlistm, xticks, xticklabels)
    print('Run time: ' + str(time.process_time() - start))
    
    