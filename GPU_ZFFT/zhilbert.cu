//#include <zfft_kernel.cu>

cufftComplex* ZHILBERT(cufftComplex *a_complex_d, cufftComplex *c_complex_d, float *kz_d, int Ny, int Nx, int Nz)
{
  
  
  float scaler;
  cudaMalloc((void**) &scaler, sizeof(float));
  
  // int threadsPerBlock = 512;
    int block_size_x=2; int block_size_y=2; 
    int dimGridx, dimGridy;

    dim3 dimBlock(block_size_x, block_size_y);
    if(Nx/dimBlock.x == 0) {dimGridx = 1;}
    else dimGridx = Nx/dimBlock.x;
    if(Ny/dimBlock.y == 0) {dimGridy = 1;}
    else dimGridy = Ny/dimBlock.y;
    dim3 dimGrid(dimGridx, dimGridy);  
    
  //dim3 dimGrid(100,50);
  //dim3 dimBlock(8,8,8);
  
  cufftHandle plan;
  int n[1] = {Nz};
  int inembed[1] = {(Ny/2+1)*Nx*Nz};
  int onembed[1] = {(Ny/2+1)*Nx*(Nz)};
  cufftPlanMany(&plan,1,n,inembed,(Ny/2+1)*Nx,1,
                          onembed,(Ny/2+1)*Nx,1,CUFFT_C2C,(Ny/2+1)*Nx);
              //    n rank  nembed  stride   dist
  
  cufftExecC2C(plan, a_complex_d, a_complex_d, CUFFT_FORWARD);
  
  //a_complex_z_d is a field of the form a(ky,kx,kz)
  
  
  
  zhilbert<<<dimGrid, dimBlock>>> (a_complex_d, c_complex_d,
                                kz_d, Ny, Nx, Nz);
								
  cufftExecC2C(plan, c_complex_d, c_complex_d, CUFFT_INVERSE);
  
  //now we have a field c of form c(ky,kx,z)
  
  scaler = (float)1/(Nz);
  
  scale<<<dimGrid,dimBlock>>> (c_complex_d, scaler, Ny, Nx, Nz);
  
  
  
  cufftDestroy(plan);
   
  
  
  return c_complex_d;
}  
