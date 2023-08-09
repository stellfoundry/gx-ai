import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from netCDF4 import Dataset

plt.figure(0)
for fname in sys.argv[1:]:
  data = Dataset(fname, mode='r')
  t = data.variables['time'][:]
  kx = data.variables['kx'][:]
  ky = data.variables['ky'][:]
  dky = ky[1] - ky[0]
  Akxt = data.groups['Spectra'].variables['Phi2kxt'][:,:]
  Akx = np.mean(Akxt[-int(len(t)/2):], axis=(0))/dky
  plt.plot(kx, Akx, 'o-')

plt.plot(kx, (kx/.3)**(-7/3)*np.max(Akx), '--', label=r"k^(-7/3)")
plt.xscale('log')
plt.yscale('log')
plt.ylim(top=np.max(Akx)*2)
plt.xlabel(r'$k_x \rho_i$')
plt.ylabel(r"$|\Phi|^2$")
plt.legend()
plt.show()
