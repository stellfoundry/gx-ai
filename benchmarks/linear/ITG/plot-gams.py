import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.style as style
from matplotlib.ticker import AutoMinorLocator

import os

from netCDF4 import Dataset

fig, (ax1, ax2) = plt.subplots(2)
fnames = sys.argv[1:]
kmax = None

for fname in fnames:
    # read gx data
    data = Dataset(fname, mode='r')
    t = data.variables['time'][:]
    ky = data.variables['ky'][1:]
    omegas = data.groups['Special'].variables['omega_v_time'][:,1:,0,0]
    gams = data.groups['Special'].variables['omega_v_time'][:,1:,0,1]
    omavg = np.mean(omegas[int(len(t)/2):, :], axis=0)
    gamavg = np.mean(gams[int(len(t)/2):, :], axis=0)
    ax1.plot(ky[:kmax], gamavg[:kmax], 'o', fillstyle='none')
    ax2.plot(ky[:kmax], omavg[:kmax], 'o', fillstyle='none', label='GX')

ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax2.xaxis.set_minor_locator(AutoMinorLocator())

ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())

ax1.set_xlim(left=0)
ax2.set_xlim(left=0)
ax1.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)
ax1.set_ylabel(r'$\gamma\ a / v_{ti}$')
#ax1.set_xlabel(r'$k_y \rho_i$')
ax2.set_ylabel(r'$\omega\ a / v_{ti}$')
ax2.set_xlabel(r'$k_y \rho_i$')
ax2.legend()
plt.tight_layout()
plt.show()
