void ZDERIV(cufftComplex *result, cufftComplex* f, float* kz) 
{
  float scaler;
  cudaMalloc((void**) &scaler, sizeof(float));
  
  int dev;
    struct cudaDeviceProp prop;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop,dev);
    int zThreads = prop.maxThreadsDim[2];
    int totalThreads = prop.maxThreadsPerBlock;     
    
    int xy = totalThreads/Nz;
    int blockxy = sqrt(xy);
    //dimBlock = threadsPerBlock, dimGrid = numBlocks
    dim3 dimBlock(blockxy,blockxy,Nz);
    if(Nz>zThreads) {
      dimBlock.x = sqrt(totalThreads/zThreads);
      dimBlock.y = sqrt(totalThreads/zThreads);
      dimBlock.z = zThreads;
    }  
    
    dim3 dimGrid(Nx/dimBlock.x+1,Ny/dimBlock.y+1,1);
  
  cufftHandle plan;
  int n[1] = {Nz};
  int inembed[1] = {(Ny/2+1)*Nx*Nz};
  int onembed[1] = {(Ny/2+1)*Nx*(Nz)};
  cufftPlanMany(&plan,1,n,inembed,(Ny/2+1)*Nx,1,
                          onembed,(Ny/2+1)*Nx,1,CUFFT_C2C,(Ny/2+1)*Nx);
              //    n rank  nembed  stride   dist
	
  
/*   struct cover {
    //the cover object will have nCases # of cases (=nky)
    int nCases;
    //each case will have nClasses # of classes
    int nClasses; 
    //each class will have nChains # of chains
    int nChains;
    //each chain will have nLinks # of links
    int nLinks;
    //we will create a 4D array of all members of the form 
    //[link#][chain#][class#][case#]
    int members[nLinks][nChains][nClasses][nCases];
  };
  
  //4D ragged array of form:
  //[case#][class#][chain#][link#]
  int[][][][] cover;
  cover[0][0][0] = {1,3,5};
  cover[0][0][1] = {2,4};
  cover[0][1][0] = {1,3};
  //...
  
  int nCases;
  int nClasses[];
  int nChains[][];
  int nLinks[][][];
  int nCoupled;
  
  //loop over all the cases
  for(int i=0; i<nCases; i++) {
    //loop over all classes within each case
    for(int j=0; j<nClasses[i]; j++) {
      //loop over all chains within each class within each case
      for(int k=0; k<nChains[i][j]; k++) {
        nCoupled = nLinks[i][j][k];
	ztransform(nCoupled,cover[i][j][k]);
      }
    }
  }
  	
        
         */
  
  
  
  
  	      
  cufftExecC2C(plan, f, result, CUFFT_FORWARD);
  
  
  
  //f is a field of the form f(ky,kx,kz)
 
  
  zderiv<<<dimGrid, dimBlock>>> (result, kz);
  
  
  
  cufftExecC2C(plan, result, result, CUFFT_INVERSE);				
  
  
  //now we have a field result of form result(ky,kx,z)
  
  scaler = (float)1/(Nz);
  
  scale<<<dimGrid,dimBlock>>> (result, scaler);
  
  
  
  
  cufftDestroy(plan);
  
  //cudaFree(a_complex_z_d); cudaFree(b_complex_z_d);
 
}  


 
