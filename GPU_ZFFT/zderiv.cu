//#include <zfft_kernel.cu>
//void getfcnC(cufftComplex* fcn, cufftComplex* fcn_d, int Ny, int Nx, int Nz);

cufftComplex* ZDERIV(cufftComplex *a_complex_d, cufftComplex *b_complex_d, float *kz_d, int Ny, int Nx, int Nz)
{
  //cufftComplex *test_complex;
  //float *kz;
  //kz = (float*) malloc(sizeof(float)*(Nz));
  //test_complex = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny/2+1)*Nx*(Nz));
  
  
  
  float scaler;
  cudaMalloc((void**) &scaler, sizeof(float));
  
  //int threadsPerBlock = 512;
    int block_size_x=2; int block_size_y=2; 
    int dimGridx, dimGridy;

    dim3 dimBlock(block_size_x, block_size_y);
    if(Nx/dimBlock.x == 0) {dimGridx = 1;}
    else dimGridx = Nx/dimBlock.x;
    if(Ny/dimBlock.y == 0) {dimGridy = 1;}
    else dimGridy = Ny/dimBlock.y;
    dim3 dimGrid(dimGridx, dimGridy);  
    
  //dim3 dimGrid(1,1);
  //dim3 dimBlock(8,8,8);
  
  
  cufftHandle plan;
  int n[1] = {Nz};
  int inembed[1] = {(Ny/2+1)*Nx*Nz};
  int onembed[1] = {(Ny/2+1)*Nx*(Nz)};
  cufftPlanMany(&plan,1,n,inembed,(Ny/2+1)*Nx,1,
                          onembed,(Ny/2+1)*Nx,1,CUFFT_C2C,(Ny/2+1)*Nx);
              //    n rank  nembed  stride   dist
	      
  cufftExecC2C(plan, a_complex_d, a_complex_d, CUFFT_FORWARD);
  
   /*printf("a(ky,kx,kz)\n");
  getfcnC(test_complex, a_complex_d, Ny, Nx, Nz);
  printf("\n"); */  
  
  //a_complex_z_d is a field of the form a(ky,kx,kz)
  
  
  
  
  zderiv<<<dimGrid, dimBlock>>> (a_complex_d, b_complex_d, kz_d, Ny, Nx, Nz);

  /* cudaMemcpy(kz, kz_d, sizeof(float)*(Nz),cudaMemcpyDeviceToHost);
  for(int i=0; i<Nz; i++) {
    printf("%f  %d\n", kz[i], i);
  } */
  				
  /*  printf("b(ky,kx,kz)\n");
  getfcnC(test_complex, b_complex_d, Ny, Nx, Nz);
  printf("\n");  */
  
  cufftExecC2C(plan, b_complex_d, b_complex_d, CUFFT_INVERSE);				
  
  //now we have a field b of form b(ky,kx,z)
  
  scaler = (float)1/(Nz);
  
  scale<<<dimGrid,dimBlock>>> (b_complex_d, scaler, Ny, Nx, Nz);
  
 
  
  
  
  cufftDestroy(plan);
  
  //cudaFree(a_complex_z_d); cudaFree(b_complex_z_d);
  
  
  return b_complex_d;
}  


 
