//for each class
void zTransformCovering(int nLinks, int nChains, int* ky, int* kx, cufftComplex* f) {
  
  cufftComplex* g;
  cudaMalloc((void**) &g, sizeof(cufftComplex)*(Nz*nLinks*nChains));
  float* kz_covering;
  cudaMalloc((void**) &kz_covering, sizeof(float)*Nz*nLinks);
  
  int dev;
  struct cudaDeviceProp prop;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop,dev);
  int zThreads = prop.maxThreadsDim[2];
  int totalThreads = prop.maxThreadsPerBlock; 
  int xy = totalThreads/nLinks;
  int blockxy = sqrt(xy);  
  dim3 dimBlock(blockxy,blockxy,nLinks);
  if(nLinks>zThreads) {
    dimBlock.x = sqrt(totalThreads/zThreads);
    dimBlock.y = sqrt(totalThreads/zThreads);
    dimBlock.z = zThreads;
  }    
    
  dim3 dimGrid(Nz/dimBlock.x+1,nChains/dimBlock.y+1,1);
  
  //also set up normal configuration
  xy = totalThreads/Nz;
  blockxy = sqrt(xy);
  dim3 dimBlock2(blockxy,blockxy,Nz);
  if(Nz>zThreads) {
    dimBlock2.x = sqrt(totalThreads/zThreads);
    dimBlock2.y = sqrt(totalThreads/zThreads);
    dimBlock2.z = zThreads;
  }  
    
  dim3 dimGrid2(Nx/dimBlock2.x+1,Ny/dimBlock2.y+1,1);
  
  //if(periodic)
  //if(noflip)
  zeroCovering<<<dimGrid,dimBlock>>>(g, nLinks, nChains);
  
  coveringCopy<<<dimGrid,dimBlock>>> (g, nLinks, nChains, ky, kx, f);
  
  mask<<<dimGrid2,dimBlock2>>>(f);
  
  cufftHandle plan;
  //use planMany, nChains
  int n[1] = {nLinks*Nz};
  cufftPlanMany(&plan,1,n,NULL,1,0,NULL,1,0,CUFFT_C2C,nChains);
  
    
  kzInitCovering<<<dimGrid,dimBlock>>>(kz_covering, nLinks);
  //kz' = kz/nCoupled
  
  cufftExecC2C(plan, g, g, CUFFT_FORWARD);
  
  //printf("A\n");
  //getfcnZCOMPLEX(g);
  
  
  zderiv_covering<<<dimGrid, dimBlock>>> (g, nLinks, nChains, kz_covering);
  
  //mask_Z_covering<<<dimGrid,dimBlock>>>(g);
  
  cufftExecC2C(plan, g, g, CUFFT_INVERSE);
  
  float scaler = (float) 1/(Nz*nLinks);
  
  scale_covering<<<dimGrid,dimBlock>>> (g, nLinks, nChains, scaler);
  
  zeroC<<<dimGrid2,dimBlock2>>>(f);
      
  coveringCopyBack<<<dimGrid,dimBlock>>> (f, nLinks, nChains, ky, kx, g);
  
  printf("F: %d\n", nLinks);
  getfcn(f);
  
  mask<<<dimGrid2,dimBlock2>>>(f);
  
  //printf("\nG':\n");  
  //getfcn(g);
  //printf("\nF':\n");
  //getfcn(f);
  
  cufftDestroy(plan);
  
  cudaFree(g); cudaFree(kz_covering);
  

}  

  
