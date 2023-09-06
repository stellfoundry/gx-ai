import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from netCDF4 import Dataset

plt.figure(0)
maxP = 0
for fname in sys.argv[1:]:
  data = Dataset(fname, mode='r')
  t = data.variables['time'][:]
  ky = data.variables['ky'][:]
  Pkyt = data.groups['Spectra'].variables['Phi2kyt'][:,:]
  Pky = np.mean(Pkyt[-int(len(t)/2):], axis=0)
  maxP = max(maxP, np.max(Pky[1:]))
  plt.plot(ky, Pky, 'o-', label=fname)

plt.xscale('log')
plt.yscale('log')
plt.plot(ky, (ky/.3)**(-7/3)*maxP, 'k--', label=r"k^(-7/3)")
plt.gca().set_ylim(top = 2*maxP)
plt.xlabel(r'$k_y \rho_i$')
plt.ylabel(r"$|\Phi|^2$")
plt.legend()
plt.show()
