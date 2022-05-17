# module to compute and plot time-average heat flux
# can be imported into another script or run as a standalone script with
# > python heat_flux.py [list of .nc files]

import numpy as np
import matplotlib.pyplot as plt
import sys

from netCDF4 import Dataset

def heat_flux(fname, ispec=0, navgfac=0.5, tag=None, plot=True, fig=None, Lref="a", refsp=None):
    # read data from file
    data = Dataset(fname, mode='r')
    t = data.variables['time'][:]
    q = data.groups['Fluxes'].variables['qflux'][:,ispec]
    species_type = data.groups['Inputs'].groups['Species'].variables['species_type'][ispec]
    if species_type == 0:
        species_tag = "i"
    elif species_type == 1:
        species_tag = "e"
    if refsp == None:
        refsp = species_tag

    # compute time-average and std dev
    istart_avg = int(len(t)*navgfac)
    qavg = np.mean(q[istart_avg:])
    qstd = np.std(q[istart_avg:])
    if tag == None:
        tag = fname
    print("%s: Q_%s/Q_GB = %.5g +/- %.5g" % (tag, species_tag, qavg, qstd))

    # make a Q vs time plot
    if plot:
        if fig == None:
            fig = plt.figure(0)
        plt.plot(t,q,'-',label="%s: Q = %.5g"%(tag, qavg))
        plt.xlim(0)
        plt.ylim(0)
        plt.ylabel("$Q_%s/Q_{GB}$"%species_tag)
        plt.xlabel("$t\ (v_{t%s}/%s)$"%(refsp, Lref))
        plt.legend()
        plt.tight_layout()

if __name__ == "__main__":
    
    for fname in sys.argv[1:]:
    
        try:
            heat_flux(fname)
        
        except:
            print(' usage: python heat-flux.py [list of .nc files]')
    
    #plt.savefig("heat_flux.png")
    plt.show()
