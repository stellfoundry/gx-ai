//for each class
inline void ZTransformCovering(int nLinks, int nChains, int* ky, int* kx, cuComplex* f,cuComplex* result, cuComplex* g, float* kz_covering, char* abs, cufftHandle plan, cudaStream_t stream, dim3 dimGridCovering, dim3 dimBlockCovering) {
    
  
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
 
 
  if(nLinks == 1) {
    reality_covering<<<dimGridCovering, dimBlockCovering,0,stream>>> (g);
  }   

  cufftExecC2C(plan, g, g, CUFFT_INVERSE);
  
  
  float scaler = (float) 1./(Nz*nLinks);
  
  
  scale_covering<<<dimGridCovering,dimBlockCovering,0,stream>>> (g, nLinks, nChains, scaler);
  
    
  //only a partial copy... don't zero f before this!    
  coveringCopyBack<<<dimGridCovering,dimBlockCovering,0,stream>>> (result, nLinks, nChains, ky, kx, g);
  
  
  
 
}  

  
