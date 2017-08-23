from numpy.polynomial import laguerre
from scipy import special
import numpy as np

def laguerre_quadrature(L):
  J = (3*L-1)//2
  roots, weights = laguerre.laggauss(J+1)
  
  toGrid = np.zeros((J+1)*(L+1))
  toSpectral = np.zeros((L+1)*(J+1))
  for j in np.arange(0,J+1):
      for l in np.arange(0,L+1):
          toGrid[j+(J+1)*l] = (-1)**l*special.eval_laguerre(l, roots[j])*np.exp(-roots[j])
          toSpectral[l+(L+1)*j] = (-1)**l*special.eval_laguerre(l, roots[j])*weights[j]

  #print "toGrid_py:"
  #for j in np.arange(0,J+1):
  #    for l in np.arange(0,L+1):
  #        print "%.4e\t"%toGrid[j+(J+1)*l],
  #    print
  #print "toSpectral_py:"
  #for j in np.arange(0,J+1):
  #    for l in np.arange(0,L+1):
  #        print "%.4e\t"%toSpectral[l+(L+1)*j],
  #    print

  toGrid = toGrid.astype(np.float32)
  toSpectral = toSpectral.astype(np.float32)
  #print toGrid.shape
  #print toGrid.dtype

  return toGrid, toSpectral

