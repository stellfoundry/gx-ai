void ZDERIVcovering(cufftComplex *result, cufftComplex* f, int** kx, int** ky, int nClasses,int* nLinks, int* nChains)
{
  //copy f to result so that f is preserved through routine
  cudaMemcpy(result, f, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
        
  for(int c=0; c<nClasses; c++) {
    zTransformCovering(nLinks[c], nChains[c], ky[c], kx[c], result);
  }    	  
  
  
 
} 


