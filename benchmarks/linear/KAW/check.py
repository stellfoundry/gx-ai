# Script to check and plot |Phi|**2 vs time for kinetic alfven wave against "correct" restults
# Usage:
# > python check.py [input file stem]
# Example:
# > python check.py kaw_betahat10.0_kp0.01
# This will check kaw_betahat10.0_kp0.01.nc against kaw_betahat10.0_kp0.01_correct.nc
# The script will print to screen the integrated relative difference in |Phi|**2,
# and also generate a plot comparing the results.

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sys

from netCDF4 import Dataset

stem = sys.argv[1]

data = Dataset("%s.nc" % stem, mode='r')
t = data.variables['time'][:]
ky = data.variables['ky'][1]
Pky = data.groups['Spectra'].variables['Pkyst'][:,0,1]
plt.plot(t, Pky, label='GX')

check = Dataset("%s_correct.nc" % stem, mode='r')
t = check.variables['time'][:]
ky = check.variables['ky'][1]
check_Pky = check.groups['Spectra'].variables['Pkyst'][:,0,1]
plt.plot(t, check_Pky, label='correct')

dt = t[1]-t[0]
diff = np.abs(Pky - check_Pky)/check_Pky
diffsum = sum(diff*dt)
if ( diffsum < 1e-3 ):
    checkstr = "TEST PASSES\n"
else:
    checkstr = "TEST FAILS\n"

print("\nIntegrated relative difference in |Phi|**2 vs time = %e... %s" % (diffsum, checkstr))

#fit = stats.linregress(t, np.log(y))
#plt.plot(t, np.exp(fit.intercept + fit.slope*t), '--', label=r'%s, ky = %.3f: $\gamma = $%.5f' % (fname, ky[i], fit.slope/2))
#print("%s, ky = %f: gamma = %f" %( fname, ky[i], fit.slope/2))

plt.yscale('log')
plt.legend()
plt.show()
