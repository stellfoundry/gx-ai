//for each class
void zTransformCovering(int nLinks, int nChains, int* ky, int* kx, cufftComplex* f) {
  
  cufftComplex* g;
  cudaMalloc((void**) &g, sizeof(cufftComplex)*(Nz*nLinks*nChains));
  float* kz_covering;
  cudaMalloc((void**) &kz_covering, sizeof(float)*Nz*nLinks);
  
  
  int xy = totalThreads/nLinks;
  int blockxy = sqrt(xy);  
  dim3 dimBlockCovering(blockxy,blockxy,nLinks);
  if(nLinks>zThreads) {
    dimBlockCovering.x = sqrt(totalThreads/zThreads);
    dimBlockCovering.y = sqrt(totalThreads/zThreads);
    dimBlockCovering.z = zThreads;
  }    
    
  dim3 dimGridCovering(Nz/dimBlockCovering.x+1,nChains/dimBlockCovering.y+1,1);
  
  
  //if(periodic)
  //if(noflip)
  zeroCovering<<<dimGridCovering,dimBlockCovering>>>(g, nLinks, nChains);
  
  coveringCopy<<<dimGridCovering,dimBlockCovering>>> (g, nLinks, nChains, ky, kx, f);
  
  mask<<<dimGrid,dimBlock>>>(f);
  
  cufftHandle plan;
  //use planMany, nChains
  int n[1] = {nLinks*Nz};
  cufftPlanMany(&plan,1,n,NULL,1,0,NULL,1,0,CUFFT_C2C,nChains);
  
    
  kzInitCovering<<<dimGridCovering,dimBlockCovering>>>(kz_covering, nLinks);
  //kz' = kz/nCoupled
  
  cufftExecC2C(plan, g, g, CUFFT_FORWARD);
  
  //printf("A\n");
  //getfcnZCOMPLEX(g);
  
  
  zderiv_covering<<<dimGridCovering, dimBlockCovering>>> (g, nLinks, nChains, kz_covering);
  
  //mask_Z_covering<<<dimGridCovering,dimBlockCovering>>>(g);
  
  cufftExecC2C(plan, g, g, CUFFT_INVERSE);
  
  float scaler = (float) 1/(Nz*nLinks);
  
  scale_covering<<<dimGridCovering,dimBlockCovering>>> (g, nLinks, nChains, scaler);
  
  zeroC<<<dimGrid,dimBlock>>>(f);
      
  coveringCopyBack<<<dimGridCovering,dimBlockCovering>>> (f, nLinks, nChains, ky, kx, g);
  
  
  mask<<<dimGrid,dimBlock>>>(f);
  
  //printf("\nG':\n");  
  //getfcn(g);
  //printf("\nF':\n");
  //getfcn(f);
  
  cufftDestroy(plan);
  
  cudaFree(g); cudaFree(kz_covering);
  

}  

  
