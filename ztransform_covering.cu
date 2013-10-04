//for each class
void ZTransformCovering(int nLinks, int nChains, int* ky, int* kx, cuComplex* f,cuComplex* result, cuComplex* g, float* kz_covering, char* abs, cufftHandle plan, cudaStream_t stream) {
    
  cufftSetStream(plan,stream);
  dim3 dimBlockCovering;
  dim3 dimGridCovering;
  
  int dev;
  struct cudaDeviceProp prop;
  cudaGetDevice(&dev);
  cudaGetDeviceProperties(&prop,dev);  
  int zBlockThreads = prop.maxThreadsDim[2];
  
  if(nLinks>zBlockThreads) dimBlockCovering.z = zBlockThreads;
  else dimBlockCovering.z = nLinks;
  int xy = totalThreads/dimBlockCovering.z;
  int blockxy = (int) sqrt(xy);  
  dimBlockCovering.x = blockxy;
  dimBlockCovering.y = blockxy;
  
  if(nLinks>zThreads) {
    dimBlockCovering.x = (int) sqrt(totalThreads/zBlockThreads);
    dimBlockCovering.y = (int) sqrt(totalThreads/zBlockThreads);
    dimBlockCovering.z = zBlockThreads;
  }    
  
  //for dirac
  if(prop.maxGridSize[2] != 1) {
    dimBlockCovering.x = 8;
    dimBlockCovering.y = 8;
    dimBlockCovering.z = 8;
  }
    
  dimGridCovering.x = Nz/dimBlockCovering.x+1;
  dimGridCovering.y = nChains/dimBlockCovering.y+1;
  if(prop.maxGridSize[2] == 1) dimGridCovering.z = 1;
  else dimGridCovering.z = nLinks*icovering/dimBlockCovering.z+1;
  
  zeroCovering<<<dimGridCovering,dimBlockCovering,0,stream>>>(g, nLinks, nChains,icovering);
  
  //only a partial copy
  coveringCopy<<<dimGridCovering,dimBlockCovering,0,stream>>> (g, nLinks, nChains, ky, kx, f, icovering);
  
  //coveringBounds<<<dimGridCovering,dimBlockCovering,0,stream>>>(g, nLinks, nChains, ky);
  
    
  kzInitCovering<<<dimGridCovering,dimBlockCovering,0,stream>>>(kz_covering, nLinks,NO_ZDERIV_COVERING, icovering);   //NO_ZDERIV is a bool that turns on or off ZDERIV terms
  //kz' = kz/nCoupled
  
  cufftExecC2C(plan, g, g, CUFFT_FORWARD);
  
     
  if(abs=="abs") {
    zderiv_abs_covering<<<dimGridCovering, dimBlockCovering,0,stream>>> (g, nLinks, nChains, kz_covering, icovering);
  } else {
    zderiv_covering<<<dimGridCovering, dimBlockCovering,0,stream>>> (g, nLinks, nChains, kz_covering, icovering);
  }
  

  cufftExecC2C(plan, g, g, CUFFT_INVERSE);
  
  
  float scaler = (float) 1./(Nz*nLinks);
  
  
  scale_covering<<<dimGridCovering,dimBlockCovering,0,stream>>> (g, nLinks, nChains, scaler);
  
    
  //only a partial copy... don't zero f before this!    
  coveringCopyBack<<<dimGridCovering,dimBlockCovering,0,stream>>> (result, nLinks, nChains, ky, kx, g);
  
  
  
 
}  

  
