# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:39:51 2021

@author: shann
"""
import numpy as np
from scipy.constants import pi, m_e, hbar, eV, e, physical_constants
from scipy.constants import k as kB
from math import ceil
import scipy.sparse as sp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os.path

#mu = 0 #meV
alpha = 80 #meV*Angstrom
beta = -1.5 #meV
V = 10 #meV
theta = 1.4*pi/180 #rad
psi = -89.6*pi/180 #rad 
w = 10 #meV
a = 3.16 #Angstrom
aM = a/theta #Angstrom ~13 nm
Auc_Ang = 3*np.sqrt(3)/2*aM**2 #Angstrom^2
Auc_nm = Auc_Ang*1e-2
Auc_m = Auc_Ang*(1e-20)
meff = 0.5*m_e #kg

muB = physical_constants['Bohr magneton'][0]
mu0 = physical_constants['vacuum mag. permeability'][0]
# Converts m(q) units from A Angstrom^2 to muB. 1e-20 converts from Angstrom^2 to m^2
AAng_to_muB = 1e-20/muB

Gj = 4*pi/(np.sqrt(3)*a) # 1/Angstrom
GMj = Gj*theta # 1/Angstrom

R = 6 # Maximum radius of k points included in the calculation
Rgrid = 32
Ggrid = np.array([[-GMj, 0],[-GMj/2, -GMj*np.sqrt(3)/2]])
FBZgrid = 1/Rgrid*(GMj/np.sqrt(3))*np.array([[-np.sqrt(3)/2, 1/2],[-np.sqrt(3)/2,-1/2]])

paulis = np.array([
        np.array([[1+0j,0+0j],[0+0j,1+0j]]),
        np.array([[0+0j,1+0j],[1+0j,0+0j]]),
        np.array([[0+0j,0-1j],[0+1j,0j]]),
        np.array([[1+0j,0+0j],[0+0j,-1+0j]])
        ])

def nFD(energy_meV, mu, T):
    if isinstance(mu, np.ndarray):
        mu = mu.reshape(mu.shape + (1,))
        
    energy_J = (energy_meV - mu)*(eV/1000)
    if T == 0.:
        if isinstance(energy_J, np.ndarray):
            return (energy_J < 0).astype(int)
        else:
            return int(energy_J < 0)
    else:
        return 1/(np.exp(energy_J/(kB*T)) + 1)

def nFDprime(energy_meV, mu, T, tolerance = 1e-10):
    if isinstance(mu, np.ndarray):
        mu = mu.reshape(mu.shape + (1,))
    energy_J = np.array((energy_meV - mu)*(eV/1000), dtype= np.longdouble)
    if T == 0.:
        raise Exception('nFDprime not compatible with zero temperature')
    else:
        #return np.exp((energy_J-mu)/(kB*T))/(kB*T*(np.exp((energy_J-mu)/(kB*T)) + 1)**2)
        # Replaced the above (commented) expression with the one below because the exp's became problematic when T is small
        #print('nFDprime: energy_J is: {0:}'.format(energy_J))
        # UNIT IS 1/JOULES
        return -1/(2*kB*T)*1/(np.cosh(energy_J/(kB*T)) + 1)

# This may now be irrelevant because I made the nFD function usable outright
def is_occupied(energy_meV, mu, T, ndecimals = 4):
    #mu_round is rounded up to the fourth decimal place
    rounding = 10**ndecimals
    mu_round = ceil(mu*rounding)/rounding
    
    if (T == 0. and energy_meV < mu_round) or (T > 0. and np.random.uniform() < nFD(energy_meV, mu, T)):
        return True
    else:
        return False

# This class defines a k point on my grid, which is defined in terms of the two directions GM1, GM2
class kMvec:
    def __init__(self, point = np.array([0.,0.]), grid = 'G'):
        # Gcoords is a two-element array which specifies the coefficient of GM1, GM2, respectively
        self.coords = point
        # Defines two basis vectors in x,y coordinates
        if isinstance(grid, np.ndarray):
            self.basis = grid
        elif isinstance(grid, str):
            if grid == 'G':
                # Default is the Moire reciprocal vectors GM1 and GM2
                self.basis = Ggrid
            elif grid == 'FBZ':
                self.basis = FBZgrid
            else: 
                self.basis = None
                raise Exception('No basis defined for the kMvec')
        # GM1 = -|G_{Mj}|*\hat{x} 
        # GM2 = -|G_{Mj}|(1/2*\hat{x} + \sqrt{3}/2\hat{y}
    
    # The add and subtract can handle any kMvecs regardless of whether they have the same basis
    # The output is a kMvec with the same basis as the first vector (i.e. if A+B, return will be in same basis as A)
    def __add__(self, kprime):
        if np.all(self.basis == kprime.basis):
            return kMvec(self.coords + kprime.coords)
        else:
            ksum = np.array([self.get_x() + kprime.get_x(), self.get_y() + kprime.get_y()])
            #print(ksum)
            
            n = (ksum[0]*self.basis[1][1] - ksum[1]*self.basis[1][0])/(self.basis[0][0]*self.basis[1][1] - self.basis[0][1]*self.basis[1][0])
            m = (ksum[0]*self.basis[0][1] - ksum[1]*self.basis[0][0])/(self.basis[0][1]*self.basis[1][0] - self.basis[0][0]*self.basis[1][1])
            
            #print('n = ' +str(n) + ' , m = ' + str(m))
            
            return kMvec(np.array([n,m]), self.basis)
    
    def __sub__(self, kprime):
        if np.all(self.basis == kprime.basis):
            return kMvec(self.coords - kprime.coords)
        else:
            kdif = np.array([self.get_x() - kprime.get_x(), self.get_y() - kprime.get_y()])
            #print(kdif)
            
            n = (kdif[0]*self.basis[1][1] - kdif[1]*self.basis[1][0])/(self.basis[0][0]*self.basis[1][1] - self.basis[0][1]*self.basis[1][0])
            m = (kdif[0]*self.basis[0][1] - kdif[1]*self.basis[0][0])/(self.basis[0][1]*self.basis[1][0] - self.basis[0][0]*self.basis[1][1])
            
            #print('n = ' +str(n) + ' , m = ' + str(m))
            
            return kMvec(np.array([n,m]), self.basis)
    
    def __mul__(self, scalar):
        #Default is multiplication by a number
        try:
            return kMvec(scalar*self.coords)
        except (TypeError, ValueError):
            print("kMvec objects can only be multiplied by a scalar or array with shape (2,)")
    
    __rmul__ = __mul__ 
        
    def __div__(self, scalar):
        try:
            return kMvec(self.coords/scalar)
        except (TypeError, ValueError):
            print("kMvec objects can only be divided by a scalar or array with shape (2,)")
                
    def get_x(self):
        return self.coords[0]*self.basis[0][0] + self.coords[1]*self.basis[1][0]
    
    def get_y(self):
        return self.coords[0]*self.basis[0][1] + self.coords[1]*self.basis[1][1]
    
    def get_len(self):
        return np.sqrt(self.get_x()**2 + self.get_y()**2)
    
    def dot(self, kprime):
        return self.get_x()*kprime.get_x() + self.get_y()*kprime.get_y()

GammaM = kMvec(np.array([0,0]))
Kplus = kMvec(np.array([2/3,-1/3]))
Kminus = kMvec(np.array([1/3,1/3]))

#Initialize an array with all the (m,n) labels
def make_kmesh(maxL):
    lim = maxL
    mesh = np.vstack((np.mgrid[-lim:lim+1, -lim:lim+1][0].flatten(), np.mgrid[-lim:lim+1, -lim:lim+1][1].flatten())).T

    del_list = []
    for i in range(0, len(mesh)):
        if abs(np.sum(mesh[i])) > maxL:
            del_list.append(i)
    #print(del_list)
    return np.delete(mesh, del_list, axis=0)

def N_unitcells(r):
    return 1 + 3*r*(r+1)

kmesh = make_kmesh(R)
    
def hlxi(kvec, l, mu, xi=1):
    if l == 1:
        kdiff = kvec - xi*Kplus
        asign = 1
    elif l == 2:
        kdiff = kvec - xi*Kminus
        asign = -1
    else:
        raise Exception('l can only be 1 or 2')
        
    #print('hlxi(): kdiff basis is ' + str(kdiff.basis))
    return (hbar**2/(2*meff)*(1e20*1000/eV)*(kdiff.get_len()**2) - mu)*paulis[0] + asign*alpha*(kdiff.get_x()*paulis[2] - kdiff.get_y()*paulis[1]) + xi*beta*paulis[3]

def Hd_block(kvec, mu, xi=1):
    block = sp.block_diag((hlxi(kvec, 1, mu, xi), hlxi(kvec, 2, mu, xi)))
    #pcolor_mat(block, 'abs')
    return block

def Hd(k0, mu, xi=1):
    blocks = ()
    for point in kmesh:
        #print('Hd(): coords of kMvec is ' +str((kMvec(point, 'G') + k0).coords))
        blocks = blocks + (Hd_block(kMvec(point, 'G') + k0, mu, xi),) #comma necessary to ensure this is interpreted as a tuple
    return sp.block_diag(blocks).tocsr()

def HTeff(xi=1):
    block = np.kron(w/2*(paulis[1]+1j*paulis[2]), paulis[0])
#    pcolor_mat(block, 'abs')
#    pcolor_mat(block, 'imag')
    
    indlist = []
    indptr = [0]
    indctr = 0
    
    Nk = len(kmesh)
    
    for ind in range(0, Nk):
        n = kmesh[ind][0]
        m = kmesh[ind][1]
        #print('ind = ' + str(ind) + ' has coords (' + str(n) + ',' + str(m) + ')')
        ind2 = None
        ind3 = None
        
        try:
            ind2 = np.where(np.all(kmesh == np.array([n, m+xi*1]), axis=1))[0][0]
            #print('ind2 = ' + str(ind2))
        except:
            pass
        try:
            ind3 = np.where(np.all(kmesh == np.array([n-xi*1, m+xi*1]), axis=1))[0][0]
            #print('ind3 = ' + str(ind3))
        except:
            pass
        
        indlist.append(ind)
        indctr += 1
        if ind2 is not None:
            #if ind2 > ind:
            indlist.append(ind2)
            indctr += 1
        if ind3 is not None:
            #if ind3 > ind: 
            indlist.append(ind3)
            indctr += 1
                
        indptr.append(indctr)
        
    #print(indlist)
    #print(indptr)
    data = np.array([block]*len(indlist))
    upper_mat = sp.bsr_matrix((data, np.array(indlist), np.array(indptr)), shape = (4*Nk, 4*Nk))
    #pcolor_mat(upper_mat)
    return (upper_mat + upper_mat.transpose()).tocsr()

def VM():
    block = V*np.kron(np.cos(psi)*paulis[0]+1j*np.sin(psi)*paulis[3], paulis[0])
    
    indlist = []
    indptr = [0]
    indctr = 0
    
    Nk = len(kmesh)
    
    for ind in range(0, Nk):
        n = kmesh[ind][0]
        m = kmesh[ind][1]
        #print('ind = ' + str(ind) + ' has coords (' + str(n) + ',' + str(m) + ')')
        ind1 = None
        ind3 = None
        ind5 = None
        
        try:
            ind1 = np.where(np.all(kmesh == np.array([n+1, m]), axis=1))[0][0]
            #print('ind2 = ' + str(ind2))
        except:
            pass
        try:
            ind3 = np.where(np.all(kmesh == np.array([n-1, m+1]), axis=1))[0][0]
            #print('ind3 = ' + str(ind3))
        except:
            pass
        try:
            ind5 = np.where(np.all(kmesh == np.array([n, m-1]), axis=1))[0][0]
            #print('ind5 = ' + str(ind5))
        except:
            pass
        
        if ind1 is not None:
            #if ind2 > ind:
            indlist.append(ind1)
            indctr += 1
        if ind3 is not None:
            #if ind3 > ind: 
            indlist.append(ind3)
            indctr += 1
        if ind5 is not None:
            indlist.append(ind5)
            indctr += 1
                
        indptr.append(indctr)
        
    data = np.array([block]*len(indlist))
    upper_mat = sp.bsr_matrix((data, np.array(indlist), np.array(indptr)), shape = (4*Nk, 4*Nk))
    
    #pcolor_mat(upper_mat, 'imag')
    return (upper_mat + upper_mat.conjugate().transpose()).tocsr()

def Hxi(k0, mu = 0, xi = 1):
    '''
    Full Hamiltonian of the tTMD model.

    Parameters
    ----------
    k0 : kMvec
        k-space point at which to evaluate Hamiltonian.
    mu : float, optional
        Chemical potential acts as offset which sets Fermi energy to 0. The default is 0.
    xi : int, optional
        Valley index. The default is 1.

    Returns
    -------
    sparse matrix
        Matrix representation of Hamiltonian with 2 layer and 2 spin dof per k.

    '''
    return Hd(k0, mu, xi) + HTeff(xi) + VM()

def delhlxi(kvec, l, coord, xi=1):
    if l == 1:
        kdiff = kvec - xi*Kplus
        asign = 1
    elif l == 2:
        kdiff = kvec - xi*Kminus
        asign = -1
    else:
        raise Exception('l can only be 1 or 2')
        
    if coord == 'x':
        #print('kx - Kx = ' + str(kdiff.get_x()))
        return hbar**2/(meff)*(1e20*1000/eV)*kdiff.get_x()*paulis[0] + asign*alpha*paulis[2]
    elif coord == 'y':
        #print('ky - Ky = ' + str(kdiff.get_y()))
        return hbar**2/(meff)*(1e20*1000/eV)*kdiff.get_y()*paulis[0] - asign*alpha*paulis[1]
    else:
        raise Exception('Coordinate for Hk derivatives not defined')
        
def delHk_block(kvec, coord, xi=1):
    #print('kvec coords = (' + str(kvec.coords[0]) + ',' + str(kvec.coords[1]) + ')')
    block = sp.block_diag((delhlxi(kvec, 1, coord, xi), delhlxi(kvec, 2, coord, xi)))
    return block

def delHk(k0, coord='x', xi=1):
    blocks = ()
    for point in kmesh:
        blocks = blocks + (delHk_block(kMvec(point, 'G') + k0, coord, xi),)
    return sp.block_diag(blocks).tocsr()

def exp_spin(evals, evecs, nindex):
    # factor of 2 needed because for each k we have two spin, two layer dof
    Sz = sp.block_diag((paulis[3],)*(2*len(kmesh)))
    if evals[1] > evals[2]:
        print('Found swapped evals - reversing order')
        print(evals[0:4])
        evals[1], evals[2] = evals[2], evals[1]
        evecs[:,[1,2]] = evecs[:,[2,1]]
        
    return np.conj(evecs[:,nindex]) @ Sz.dot(evecs[:,nindex])

def Berry_curv(k0, nindex, evals, evecs, xi=1):
    '''
    Calculates berry curvature (BC) and orbital magnetic moment (mm), both intrinsic and edge

    Parameters
    ----------
    k0 : kMvec
        k point at which to evaluate.
    mu : float
        chemical potential.
    nindex : int
        band index.
    evals : ndarray
        1d array containing eigenvalues.
    evecs : ndarray
        2d array containing eigenvectors. evecs[:n] is the eigenvector for band n
    xi : int, optional
        Valley index. The default is 1.

    Returns
    -------
    ndarray with shape (3,)
        0th element is the berry curvature, 1st is the intrinsic mm, 2nd is edge mm.
        BC units are Angstrom^2, mm units are A*Angstrom^2

    '''
    nbands = len(evals)
    dxHk = delHk(k0, 'x', xi)
    dyHk = delHk(k0, 'y', xi)
    
    # If 2nd and 3rd Moire band get reversed, swap them
        # I only do this for 2nd and 3rd because they are closest in energy --> most likely to get swapped
        # An easy fix would simply be to call np.sort
    if evals[1] > evals[2]:
        print('Found swapped evals - reversing order')
        print(evals[0:4])
        evals[1], evals[2] = evals[2], evals[1]
        evecs[:,[1,2]] = evecs[:,[2,1]]

    # bc = Berry curvature
    Bc_tot = 0
    # m = intrinsic magnetic moment
    m_tot = 0
    # mtilde = edge state magnetic moment
    mtilde_tot = 0
    
    for mindex in range(0, nbands):
        if mindex == nindex:
            continue
        
        Bc_coeff = 1/(evals[nindex] - evals[mindex])**2
        m_coeff = e/(2*hbar)*(eV/1000)*(evals[mindex] - evals[nindex])*Bc_coeff
        
        dxdy =  (np.conj(evecs[:,nindex]) @ dxHk @ evecs[:,mindex])*(np.conj(evecs[:,mindex]) @ dyHk @ evecs[:,nindex])
        dydx =  (np.conj(evecs[:,nindex]) @ dyHk @ evecs[:,mindex])*(np.conj(evecs[:,mindex]) @ dxHk @ evecs[:,nindex])
        
        Bc_tot += Bc_coeff*(dxdy - dydx)
        m_tot += m_coeff*(dxdy - dydx)
        
    # Returns in units of Ampere*Angstrom^2
    return np.imag(np.array([-Bc_tot, m_tot]))

class BerryCurvQuantities:
    def __init__(self, Bc_array, band, mu, temp=0):
        '''
        Class which automates calculation of Berry curvature-related quantities which require a sum over the Brillouin zone.

        Parameters
        ----------
        Bc_array : ndarray
            2d array with shape (Nk, 3), where Nk is the number of k points used in FBZ mesh
            Bc_array[:,0] = berry curvature in Angstrom^2
            Bc_array[:,1] = intrinsic magnetic moment in A Angstrom^2
            
        band : int
            band index.
            
        mu : float
            Chemical potential used to calculated Bc_array.
            * Important that it matches mu used for Berry_curv function.
            
        temp : float, optional
            Temperature. The default is 0.

        Returns
        -------
        None.

        '''
        
        self._Bc_array = Bc_array
        self.band = band
        self.mus = mu
        self.temp = temp
        
    def get_Bc(self, evals):
        fe = nFD(evals, self.mus, self.temp)
        bc_list = self._Bc_array[:, 0]*fe
        
        if bc_list.ndim == 1:
            return np.sum(bc_list)/(2*pi*len(evals)*Auc_nm)
        elif bc_list.ndim == 2:
            return np.sum(bc_list, axis = 1)/(2*pi*len(evals)*Auc_nm)
        
    
    def get_Mz_int(self, evals):
        '''
        Calculates the total orbital magnetization, which is k space sum of orbital magnetic moment

        Parameters
        ----------
        evals : ndarray
            1d array with list of eigenvalues for each k for given band
            Should be indexed as evals_array[:, self.band]

        Returns
        -------
        float
            Mz in Bohr magnetons/nm^2.

        '''
        # Units: muB
        # m_list should always be 1d
        m_list = AAng_to_muB*self._Bc_array[:, 1]
        fe = nFD(evals, self.mus, self.temp)
        
        m_list = m_list*fe
        # Units: muB/nm^2
        if m_list.ndim == 1:
            return np.sum(m_list)/(len(evals)*Auc_nm)
        elif m_list.ndim == 2:
            print('M_list is 2 dimensional')
            return np.sum(m_list, axis = 1)/(len(evals)*Auc_nm)
    
    def get_Mz_edge(self, evals):
        bc_list = self._Bc_array[:, 0]
        fe = nFD(evals, self.mus, self.temp)
        
        if isinstance(self.mus, np.ndarray):
            mus_rs = self.mus.reshape(self.mus.shape + (1,))
            medge_list = (mus_rs - evals)*bc_list
        else:
            medge_list = (self.mus - evals)*bc_list
        medge_list = e/hbar*(eV/1000)*AAng_to_muB*medge_list*fe
        
        if medge_list.ndim == 1:
            return np.sum(medge_list)/(len(evals)*Auc_nm)
        elif medge_list.ndim == 2:
            return np.sum(medge_list, axis = 1)/(len(evals)*Auc_nm)
        else:
            print('Invalid number of dimensions')
        
    def get_orb_susc(self, evals, edge = True):
        m_int = AAng_to_muB*self._Bc_array[:,1] # Units: muB
        omega = self._Bc_array[:,0] # Units: Ang^2
        
        fe = nFD(evals, self.mus, self.temp)
        
        if self.temp == 0:     
            susc_list = e/(hbar/eV)*m_int*omega*AAng_to_muB*fe # Units: muB^2/eV
        else:
            susc_list = -np.power(m_int, 2)*nFDprime(evals, self.mus, self.temp)*eV # Units: muB^2/eV
            if edge:
                susc_list = susc_list + (e*eV)/hbar*m_int*omega*AAng_to_muB*fe 
        
        if susc_list.ndim == 1:
            return np.sum(susc_list)/(len(evals)*Auc_nm) # Units: muB^2/(eV nm^2)
        if susc_list.ndim == 2:
            return np.sum(susc_list, axis = 1)/(len(evals)*Auc_nm)
        
class GLFreeEnergy(BerryCurvQuantities):
    '''
    Child class of BerryCurvQuantities.
    Uses the data generated by its parents class to calculate Ginzburg-Landau theory parameters
    '''
    def __init__(self, Bc_array, evals, band, mu, temp, moment = 'total'):
        BerryCurvQuantities.__init__(self, Bc_array, band, mu, temp)
        
        orb_susc = BerryCurvQuantities.get_orb_susc(self, evals, moment)
        self._set_a(orb_susc)
        self._set_b(BerryCurvQuantities.get_Mz(self, evals), orb_susc)
    
    def _set_a(self, orb_susc):
        self.a = 1/(4*orb_susc)
        return self.a
    
    def _set_b(self, Mz, orb_susc):
        self.b = 1/(8*Mz**2*orb_susc)
        return self.b
    
    def Bcrit(self):
        return 4*self.a/3*np.sqrt(self.a/(6*self.b))*eV/muB
    
    def param_array(self):
        return np.array([self.a, self.b, self.Bcrit()])
    
def pcolor_mat(mat, value='abs', plotrange = None):
    if not isinstance(mat, np.ndarray):
        mat = mat.toarray()
    
    if value == 'abs':
        plot_arr = np.abs(mat)
    elif value == 'real':
        plot_arr = np.real(mat)
    elif value == 'imag': 
        plot_arr = np.imag(mat)
    
    try:
        plt.pcolor(plot_arr[plotrange[0]:plotrange[1], plotrange[0]:plotrange[1]])
    except:
        plt.pcolor(plot_arr)
        
    plt.colorbar()
    plt.show()
    
def kpath(contour = 'KGammaK', step = 1):
    path = [] 
    xticks = ''
    xticklabels = ''
    if contour == 'KGammaKKK':
        seg1 = []
        seg2 = []
        seg3 = []
        for i in np.arange(-Rgrid,Rgrid,step):
            seg1.append(np.array([i,0]))
        for i in np.arange(0,Rgrid,step):
            seg2.append(np.array([Rgrid-i, i]))
            seg3.append(np.array([-i, Rgrid-i]))
        path = seg1 + seg2 +seg3
        
        xticks = np.arange(0, len(path)+1, len(seg2))
        print('xticks: ' + str(xticks))
        xticklabels = np.array([r"$K'_-$", r"$\Gamma_M$", r"$K_+$", r"$K_-$", r"$K'_-$"])
        print('xticklabels: ' + str(xticklabels))
        
    if contour == 'KGammaKK':
        seg1 = []
        seg2 = []
        seg3 = []
        for i in np.arange(0, Rgrid, step):
            seg1.append(np.array([Rgrid-i,0]))
            seg2.append(np.array([0,i]))
            seg3.append(np.array([i,Rgrid-i]))
        path = seg1 + seg2 + seg3
        
        xticks = np.arange(0, len(path)+1, len(seg2))
        print('xticks: ' + str(xticks))
        xticklabels = np.array([r"$K_+$", r"$\Gamma_M$", r"$K_-$", r"$K_+$"])
        
    if contour == 'KGammaK':
        seg1 = []
        seg2 = []
        for i in range(0,Rgrid):
            seg1.append(np.array([Rgrid-i,0]))
            seg2.append(np.array([0,i]))
        path = seg1 + seg2
        
        xticks = np.arange(0, len(path)+1, len(seg2))
        print('xticks: ' + str(xticks))
        xticklabels = np.array([r"$K_+$", r"$\Gamma_M$", r"$K_-$"])
        
    return np.array(path), xticks, xticklabels

def plot_eigenlist(elist, xticks, xticklabels, pointstyle=None, ylimit = None, axhline = None, label = None):
    if pointstyle is None:
        plt.plot(elist)
    else:
        plt.plot(elist, pointstyle)
    if ylimit is not None:
        plt.ylim(ylimit[0],ylimit[1])
    if axhline is not None:
        try:
            plt.axhline(axhline, color='black', linestyle='--')
        except: 
            if isinstance(axhline, np.ndarray):
                for line in axhline:
                    plt.axhline(line, color='black', linestyle='--')
    plt.ylabel('Energy (meV)', size = 'xx-large')
    plt.xlabel('')
    plt.xticks(xticks, xticklabels, fontsize=15)
    if label is not None:
        plt.savefig('spectrum-' + label + '.png', bbox_inches='tight')
    plt.show()
    
def plot_grid(mesh, colour = None, grid = 'G', title='', clims = None, savename = None, pointsize=None, cvalue = 'mz'):
    xlist = []
    ylist = []
    
    for row in mesh:
        xlist.append(kMvec(row, grid).get_x())
        ylist.append(kMvec(row, grid).get_y())
    
    if grid == 'G':
        mark = 'h'
    elif grid == 'FBZ':
        mark = 'H'
    else:
        mark = 'o'
        
    if pointsize is not None:
        size = pointsize
    else:
        # Change size in proportion to R=8 case, which for a (9,8) size figure looks best wiht pointsize=700 (FBZ case)
        size = N_unitcells(8)*700/len(mesh)
    if colour is None:
        plt.scatter(xlist, ylist, s=size, marker=mark)
    else:
        if clims == None:
            minmax = np.max(np.abs(colour))
            plt.scatter(xlist, ylist, c=colour, cmap = 'Spectral', s=size, marker=mark, vmin=-minmax, vmax=minmax)
        else:
            plt.scatter(xlist, ylist, c=colour, cmap = 'Spectral', s=size, marker=mark, vmin=clims[0], vmax=clims[1])
    
    if colour is not None and cvalue == 'mz':
        plt.colorbar().set_label(label = r'$m_z(\mathbf{k})$ ($\mu_B$)', size=15)
    elif colour is not None and cvalue == 'Berrycurv':
        plt.colorbar().set_label(label = r'$\Omega(\mathbf{k})$ ($\mathrm{\AA}$)', size=15)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.xlabel(r'$k_x$', size='xx-large')
    plt.ylabel(r'$k_y$', size='xx-large')
    plt.title(title, size='xx-large')
    
    if savename is not None:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig('plots/' + savename)
    
    plt.show()

class SaveInfo:
    def __init__(self, band, tag=None, folder = 'data'):
        self.band = band
        self.tag = tag
        self.folder = folder
        
        if self.tag is None:
            raise Exception('SaveInfo() tag cannot be None!')
        else:
            self._set_filename()
            
    def _set_filename(self):
        self.filename = self.tag + '_band' + str(self.band) + '_R' + str(Rgrid)
    
    def save_data(self, array):
        #If file does not exist, make it and set ntrials to 1
        pathname = self.folder + '/' + self.filename + '.npz'
        if not os.path.isfile(pathname):
            print('Could not find file with name: ' + pathname + '.  Creating new path.')
            if not os.path.exists(self.folder):
                os.makedirs(self.folder)
            np.savez(pathname, data=array)
        #If file does exist, open the array and make sure that the lengths match
        # If they do, then vstack the new data and resave the file with the same name, add to ntrials
        else:
            data_old = np.load(pathname)['data']
            try:
                dsize = data_old.shape[1]
            except IndexError:
                dsize = len(data_old)
                
            try:
                asize = array.shape[1]
            except IndexError:
                asize = len(array)
                
            if dsize == asize:
                data_new = np.vstack((data_old, array))
                np.savez(pathname, data=data_new)
                    
            else:
                raise Exception('Length of input does not match file you wish to write to')

        return
    
    def load_data(self):
        return np.load(self.folder + '/' + self.filename)['data']

class muLimits:
    '''
    Class which standardizes the set of mus that we will use to plot Mz or susc vs. mu 
    '''
    def __init__(self, filename, band, rangetype = 'band'):
        '''
        Parameters
        ----------
        filename : string
            Name of file containing the band edges.
        band : int
            band index.
        rangetype : string, optional
            Tells set_lims method which range of chemical potential to use. The default is 'band'.

        Raises
        ------
        Warning
            If filename provided does not have the correct dictionary entries.

        Returns
        -------
        None.

        '''
        try:
            self.band_top = np.load(filename)['top']
        except:
            raise Warning('muLimits: Something went wrong loading the band_top array')
        try:
            self.band_bottom = np.load(filename)['bottom']
        except:
            raise Warning('muLimits: Something went wrong loading the band_botoom array')
        self.rangetype = rangetype
        self.set_lims(band)

    def set_lims(self, band):
        if self.rangetype == 'band':  # between band edges only
            self.lims = np.array([self.band_bottom[band], self.band_top[band]])
            
        elif self.rangetype == 'bandplusgap':  
            if band == 0 or band == 2: # bottom of band up to bottom of next band
                self.lims = np.array([self.band_bottom[band], self.band_bottom[band+1]])
            else:  
                # When next band is not well defined, just use double the width of the band
                self.lims = np.array([self.band_bottom[band], 2*self.band_top[band] - self.band_bottom[band]])
        elif self.rangetype == 'gap':
            # mu inside gape
            if band == 0 or band == 2:
                self.lims = np.array([self.band_top[band], self.band_bottom[band+1]])
            else:
                # When next band is not well defined, just use double the width of the band
                self.lims = np.array([self.band_top[band], 2*self.band_top[band] - self.band_bottom[band]])
        else:
            raise Warning('muLimits: None of the preset limit types were chosen')
            
    def set_rangetype(self, band, rangetype):
        if rangetype in np.array(['band', 'bandplusgap', 'gap']):
            self.set_lims(band)
        else:
            raise Warning('muLimits: You did not enter a valid range type')
        

def plot_vs_mu(mu_list, y_list, ytype = 'Mz', band_region = None, title = None, marker = None, xlims = None, ylims = None, logy = False, show = True):
    if marker is None:
        plt.plot(mu_list, y_list, linewidth = 4)
    else:
        plt.scatter(mu_list, y_list, marker=marker, s = 100)
        
    if logy:
        plt.yscale('log')
    # if band_top is not None:
    #     plt.axvline(band_top, color='black', linestyle='--')
    #     try:
    #         plt.axhline(y_list[mu_list == band_top][0], color='black', linestyle='--')
    #     except:
    #         print('mu_list does not contain element eactly equal to band_top')
    if band_region is not None:
        plt.axvspan(band_region[0], band_region[1], color='grey', alpha=0.4)
        
    plt.xlabel(r'$\mu$ (meV)', size='xx-large')
    if ytype == 'Mz':
        plt.ylabel(r'$M_z$ ($\mu_B/nm^2$)', size = 'xx-large')
    elif ytype == 'Mz_muB':
        plt.ylabel(r'$M_z$ ($\mu_B$ / Moire supercell)', size = 'xx-large')
    elif ytype == 'susc':
        plt.axhline(0, color='black', linestyle='--')
        plt.ylabel(r'$\chi_{orb}$ ($\mu_B \, eV^{-1} \, nm^{-2}$)', size = 'xx-large')
    elif ytype == 'Bc':
        plt.ylabel(r'$B_c$ (T)', size = 'xx-large')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])
    if ylims is not None:
        plt.ylim(ylims[0], ylims[1])
    plt.grid(axis='both', which='both')
    if title is not None:
        plt.title(title, size='xx-large')
    if show:
        plt.show()
    
def plot_Mz_vs_mu(mu_list, mint, medge, band_region = None, title = None, marker = None, xlims = None, ylims = None, logy = False, ytype = 'Mz'):
    if marker is None:
        plt.plot(mu_list, mint + medge, linewidth = 5, label = r'$m_z^{int} + m_z^{(\Omega)}$', color = cm.cool(1/3))
        plt.plot(mu_list, mint, '--', linewidth = 4, label = r'$m_z^{int}$', color = cm.cool(2/3))
        plt.plot(mu_list, medge, '-.', linewidth = 4, label = r'$m_z^{(\Omega)}$', color = cm.cool(3/3))
    else:
        plt.scatter(mu_list, mint + medge, marker=marker, s = 100, label = r'$m_z^{int} + m_z^{edge}$')
        plt.scatter(mu_list, mint, marker=marker, s = 100, label = r'$m_z^{int}$')
        plt.scatter(mu_list, medge, marker=marker, s = 100, label = r'$m_z^{edge}$')
        
    if ytype == 'Mz':
        plt.ylabel(r'$M_z$ ($\mu_B/nm^2$)', size = 'xx-large')
    elif ytype == 'Mz_muB':
        plt.ylabel(r'$M_z$ ($\mu_B$ / Moire supercell)', size = 'xx-large')
   
    if logy:
        plt.yscale('log')
    # if band_top is not None:
    #     plt.axvline(band_top, color='black', linestyle='--')
    #     try:
    #         plt.axhline(y_list[mu_list == band_top][0], color='black', linestyle='--')
    #     except:
    #         print('mu_list does not contain element eactly equal to band_top')
    if band_region is not None:
        plt.axvspan(band_region[0], band_region[1], color='grey', alpha=0.4)
        
    plt.xlabel(r'$\mu$ (meV)', size='xx-large')
    plt.legend(fontsize=15)

    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    if xlims is not None:
        plt.xlim(xlims[0], xlims[1])
    if ylims is not None:
        plt.ylim(ylims[0], ylims[1])
    plt.grid(axis='both', which='both')
    if title is not None:
        plt.title(title, size='xx-large')
    plt.show()
        
        
# def a_GL(chizz):
#     return 1/(4*chizz)

# def b_GL(Mz, chizz):
#     return 1/(8*Mz**2*chizz)

# def Bcrit(a, b):
#     # units eV/muB 
#     return 4*a/3*np.sqrt(a/(6*b))
    

