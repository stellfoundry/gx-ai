import numpy as np
from scipy import special

def laguerre_quadrature(L, printout=False):
  J = (3*L-1)//2
  roots, weights = special.roots_laguerre(J+1)

  toSpectral = np.zeros((J+1,L+1))
  toGrid = np.zeros((L+1,J+1))
  
  for j in np.arange(0,J+1):
      for l in np.arange(0,L+1):
          toGrid[l,j] = (-1)**l*special.eval_laguerre(l, roots[j])
          toSpectral[j,l] = (-1)**l*special.eval_laguerre(l, roots[j])*weights[j]
  
  toGrid = toGrid.astype(np.float32).flatten('F')
  toSpectral = toSpectral.astype(np.float32).flatten('F')
  roots = roots.astype(np.float32)

  if printout:
      print "toGrid_py:"
      print toGrid
      print "toSpectral_py:"
      print toSpectral
      print "roots_py:"
      print roots
      print
      
  return toGrid, toSpectral, roots

