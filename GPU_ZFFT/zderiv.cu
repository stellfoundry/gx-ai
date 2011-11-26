//#include <zfft_kernel.cu>
//void getfcnC(cufftComplex* fcn, cufftComplex* fcn_d, int Ny, int Nx, int Nz);

cufftComplex* ZDERIV(cufftComplex *a_complex_d, cufftComplex *b_complex_d, float *kz_d, int Ny, int Nx, int Nz)
{ 
  float scaler;
  cudaMalloc((void**) &scaler, sizeof(float));
  
  int xy = 512/Nz;
  int blockxy = sqrt(xy);
  //dimBlock = threadsPerBlock, dimGrid = numBlocks
  dim3 dimBlock(blockxy,blockxy,Nz);
  dim3 dimGrid(Nx/dimBlock.x+1,Ny/dimBlock.y+1,1);
  
  cufftHandle plan;
  int n[1] = {Nz};
  int inembed[1] = {(Ny/2+1)*Nx*Nz};
  int onembed[1] = {(Ny/2+1)*Nx*(Nz)};
  cufftPlanMany(&plan,1,n,inembed,(Ny/2+1)*Nx,1,
                          onembed,(Ny/2+1)*Nx,1,CUFFT_C2C,(Ny/2+1)*Nx);
              //    n rank  nembed  stride   dist
	      
  cufftExecC2C(plan, a_complex_d, a_complex_d, CUFFT_FORWARD);
  
 
  
  //a_complex_z_d is a field of the form a(ky,kx,kz)
 
  
  zderiv<<<dimGrid, dimBlock>>> (a_complex_d, b_complex_d, kz_d, Ny, Nx, Nz);

  
  cufftExecC2C(plan, b_complex_d, b_complex_d, CUFFT_INVERSE);				
  
  //now we have a field b of form b(ky,kx,z)
  
  scaler = (float)1/(Nz);
  
  scale<<<dimGrid,dimBlock>>> (b_complex_d, scaler, Ny, Nx, Nz);
  
  
  cufftDestroy(plan);
  
  //cudaFree(a_complex_z_d); cudaFree(b_complex_z_d);
  
  
  return b_complex_d;
}  


 
