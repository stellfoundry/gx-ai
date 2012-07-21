//for each class
void zTransformCovering(int nLinks, int nChains, int* ky, int* kx, int* ky_h, int* kx_h, cufftComplex* f) {
    
  //printf("\n\n\nnLinks = %d\n\n\n",nLinks);
  
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
  
  //kPrint(nLinks,nChains,ky_h,kx_h);
  
  //if(periodic)
  //if(noflip)
  zeroCovering<<<dimGridCovering,dimBlockCovering>>>(g, nLinks, nChains);
  
  //only a partial copy
  coveringCopy<<<dimGridCovering,dimBlockCovering>>> (g, nLinks, nChains, ky, kx, f);
   
   
  mask<<<dimGrid,dimBlock>>>(f);
  
  //getfcn_Covering(g,nLinks,nChains,ky_h,kx_h);
  
  cufftHandle plan;
  //use planMany, nChains
  int n[1] = {nLinks*Nz};
  cufftPlanMany(&plan,1,n,NULL,1,0,NULL,1,0,CUFFT_C2C,nChains);
  
    
  kzInitCovering<<<dimGridCovering,dimBlockCovering>>>(kz_covering, nLinks);
  //kz' = kz/nCoupled
  
  cufftExecC2C(plan, g, g, CUFFT_FORWARD);
  
  float scaler2 = (float) 2/(Nz*nLinks);
  
  scale_covering<<<dimGridCovering,dimBlockCovering>>>(g, nLinks, nChains, scaler2);
  
  
  //getfcnZCOMPLEX_Covering(g,nLinks,nChains,ky_h,kx_h,kz_covering); 
     
  
  zderiv_covering<<<dimGridCovering, dimBlockCovering>>> (g, nLinks, nChains, kz_covering);

  mask_Z_covering<<<dimGridCovering,dimBlockCovering>>>(g,nLinks,nChains); 
  
  //getfcnZCOMPLEX_Covering(g,nLinks,nChains,ky_h,kx_h,kz_covering);  
  
  cufftExecC2C(plan, g, g, CUFFT_INVERSE);
  
  float scaler = (float) 1/2;//1/(Nz*nLinks);
  
  scale_covering<<<dimGridCovering,dimBlockCovering>>> (g, nLinks, nChains, scaler);
  
  //getfcn_Covering(g,nLinks,nChains,ky_h,kx_h);
    
  //only a partial copy... don't zero f before this!    
  coveringCopyBack<<<dimGridCovering,dimBlockCovering>>> (f, nLinks, nChains, ky, kx, g);
  
  
  mask<<<dimGrid,dimBlock>>>(f);
  
  //printf("\nG':\n");  
  //getfcn(g);
  //printf("\nF':\n");
  //getfcn(f);
  
  cufftDestroy(plan);
  
  cudaFree(g); cudaFree(kz_covering);
  

}  

  
