//for each class
void zTransformCovering(int nLinks, int nChains, int* ky, int* kx, cufftComplex* f) {
  
  cufftComplex* g;
  cudaMalloc((void**) &g, sizeof(cufftComplex)*(2*Nz*nLinks*nChains));
  float* kz_covering;
  cudaMalloc((void**) &kz_covering, sizeof(float)*2*Nz*nLinks);
  
  int dev;
  struct cudaDeviceProp prop;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop,dev);
  int totalThreads = prop.maxThreadsPerBlock; 
  int xthreads = totalThreads/nLinks/nChains;  
  dim3 dimBlock(xthreads,nLinks,nChains);
  dim3 dimGrid(2*Nz/dimBlock.x+1,nLinks/dimBlock.y+1,1);
  
  
  
  coveringCopy<<<dimGrid,dimBlock>>> (g, nLinks, nChains, ky, kx, f);
  
  cufftHandle plan;
  //use planMany, nChains
  int n[1] = {2*nLinks*Nz};
  cufftPlanMany(&plan,1,n,NULL,1,0,NULL,1,0,CUFFT_C2C,nChains);
  
    
  kzInitCovering<<<dimGrid,dimBlock>>>(kz_covering, nLinks);
  //kz' = kz/nCoupled
  
  cufftExecC2C(plan, g, g, CUFFT_FORWARD);
  
  zderiv_covering<<<dimGrid, dimBlock>>> (g, nLinks, nChains, kz_covering);
  
  cufftExecC2C(plan, g, g, CUFFT_INVERSE);
  
  float scaler = (float) 1/(2*Nz*nLinks);
  
  scale_covering<<<dimGrid,dimBlock>>> (g, nLinks, nChains, scaler);
      
  coveringCopyBack<<<dimGrid,dimBlock>>> (f, nLinks, nChains, ky, kx, g);
  
  cufftDestroy(plan);
  

}  

  
