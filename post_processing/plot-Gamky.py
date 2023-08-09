import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from netCDF4 import Dataset

plt.figure(0)
for fname in sys.argv[1:]:
  data = Dataset(fname, mode='r')
  t = data.variables['time'][:]
  ky = data.variables['ky'][:]
  Gamkyt = data.groups['Spectra'].variables['Gamkyst'][:,0,:]
  Gamky = np.mean(Gamkyt[-int(len(t)/2):], axis=0)
  plt.plot(ky, Gamky, 'o-')

refsp = 'i'
plt.xlabel(r'$k_y \rho_{%s}$' % refsp)
plt.ylabel(r"$\Gamma/\Gamma_\mathrm{GB}$")
#plt.xscale('log')
#plt.gca().set_yscale('symlog')
plt.tight_layout()
plt.show()
