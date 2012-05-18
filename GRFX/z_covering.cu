void ZDERIVcovering(cufftComplex *result, cufftComplex* f, int** kx, int** ky, int nClasses,int* nLinks, int* nChains)
{
  //copy f to result so that f is preserved through routine
  cudaMemcpy(result, f, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz, cudaMemcpyDeviceToDevice);
  
  int **kxCover_h, **kyCover_h;
  kxCover_h = (int**) malloc(sizeof(int)*nClasses);
  kyCover_h = (int**) malloc(sizeof(int)*nClasses); 
        
  for(int c=0; c<nClasses; c++) {      
    kyCover_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
    kxCover_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]); 
    cudaMemcpy(kyCover_h[c], ky[c], sizeof(int)*nLinks[c]*nChains[c], cudaMemcpyDeviceToHost);
    cudaMemcpy(kxCover_h[c], kx[c], sizeof(int)*nLinks[c]*nChains[c], cudaMemcpyDeviceToHost);
    //kPrint(nLinks[c], nChains[c],kyCover_h[c],kxCover_h[c]);
    zTransformCovering(nLinks[c], nChains[c], ky[c], kx[c], kyCover_h[c],kxCover_h[c],result);
    free(kxCover_h[c]); free(kyCover_h[c]);
  }    	  
  
  
  
 
} 


