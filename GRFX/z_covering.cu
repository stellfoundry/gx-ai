void ZDERIVcovering(cufftComplex *result, cufftComplex* f)
{
  //initialize the kx and ky arrays for the covering space  
  int nClasses;
  int* nLinks;
  cudaMalloc((void**) &nLinks, sizeof(int)*nClasses);
  int* nChains;
  cudaMalloc((void**) &nChains, sizeof(int)*nClasses);
  
  //each class has nChains[c] chains, all with the same length, nLinks[c]
  
  int *ky[nClasses];
  int *kx[nClasses];
  for(int c=0; c<nClasses; c++) {
    cudaMalloc((void**) &ky[c], sizeof(int)*nLinks[c]*nChains[c]);
    cudaMalloc((void**) &kx[c], sizeof(int)*nLinks[c]*nChains[c]);
  }  
      
    
  
  for(int c=0; c<nClasses; c++) {
    zTransformCovering(nLinks[c], nChains[c], ky[c], kx[c], f);
  }    	

 
}  
