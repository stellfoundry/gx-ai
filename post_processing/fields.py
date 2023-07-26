import pdb
import os
import sys
import numpy as np
from netCDF4 import Dataset
from matplotlib import pyplot as plt

for i, fname in enumerate(sys.argv[1:-1]):
    iky = int(sys.argv[-1])

    rtg = Dataset(fname, 'r')
    # read from eiktest output
    tgrid = rtg.variables['theta'][:].data
    nz = tgrid.shape[0]
    ky = rtg.variables['ky'][:].data

    if i==0:
        nFields = 1
    try:
        phi  = rtg.groups['Special'].variables['Phi_z'][:].data[iky,0,:,:]
        phiC = phi[:,0] + 1j*phi[:,1]
        norm = phiC[int(nz/2)]
        phi  = np.transpose(np.array([(phiC /norm).real, (phiC /norm).imag]))
    except:
        print("No fields have been written! Set fields=true in the [Diagnostics] section of the input file")
        exit(1)
    try:
        apar = rtg.groups['Special'].variables['Apar_z'][:].data[iky,0,:,:]
        aparC = apar[:,0] + 1j*apar[:,1]
        apar = np.transpose(np.array([(aparC/norm).real, (aparC/norm).imag]))
        hasApar = True
        if i==0:
            nFields += 1
    except:
        hasApar = False
    try:
        bpar = rtg.groups['Special'].variables['Bpar_z'][:].data[iky,0,:,:]
        bparC = bpar[:,0] + 1j*bpar[:,1]
        bpar = np.transpose(np.array([(bparC/norm).real, (bparC/norm).imag]))
        hasBpar = True
        if i==0:
            nFields += 1
    except:
        hasBpar = False

    if nFields==1:
        if i==0:
            fig, ax = plt.subplots(1,1, figsize=(5,5))
        ax.plot(tgrid, phi[:,0], '-', color=f'C{i}')
        ax.plot(tgrid, phi[:,1], '--', color=f'C{i}')
        ax.set_title(r'$e\Phi/T_\mathrm{ref}$')
    if nFields==2:
        if i==0:
            fig, ax = plt.subplots(1,2, figsize=(10,5))
        ax[0].plot(tgrid, phi[:,0], '-', color=f'C{i}')
        ax[0].plot(tgrid, phi[:,1], '--', color=f'C{i}')
        ax[0].set_title(r'$e\Phi/T_\mathrm{ref}$')
        if hasApar:
            ax[1].plot(tgrid, apar[:,0], '-', color=f'C{i}')
            ax[1].plot(tgrid, apar[:,1], '--', color=f'C{i}')
            ax[1].set_title(r'$A_\parallel/(\rho_\mathrm{ref} B_N)$')
        if hasBpar:
            ax[1].plot(tgrid, bpar[:,0], '-', color=f'C{i}')
            ax[1].plot(tgrid, bpar[:,1], '--', color=f'C{i}')
            ax[1].set_title(r'$\delta B_\parallel/B$')
    if nFields==3:
        if i==0:
            fig, ax = plt.subplots(2,2, figsize=(10,10))
        ax[0,0].plot(tgrid, phi[:,0], '-', color=f'C{i}')
        ax[0,0].plot(tgrid, phi[:,1], '--', color=f'C{i}')
        ax[0,0].set_title(r'$e\Phi/T_\mathrm{ref}$')
        if hasApar:
            ax[0,1].plot(tgrid, apar[:,0], '-', color=f'C{i}')
            ax[0,1].plot(tgrid, apar[:,1], '--', color=f'C{i}')
            ax[0,1].set_title(r'$A_\parallel/(\rho_\mathrm{ref} B_N)$')
        if hasBpar:
            ax[1,0].plot(tgrid, bpar[:,0], '-', color=f'C{i}')
            ax[1,0].plot(tgrid, bpar[:,1], '--', color=f'C{i}')
            ax[1,0].set_title(r'$\delta B_\parallel/B$')

plt.show()

