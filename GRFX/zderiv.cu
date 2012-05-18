void ZDERIV(cufftComplex *result, cufftComplex* f, float* kz) 
{
  float scaler;
  //cudaMalloc((void**) &scaler, sizeof(float));
  
  
  cufftHandle plan;
  int n[1] = {Nz};
  int inembed[1] = {(Ny/2+1)*Nx*Nz};
  int onembed[1] = {(Ny/2+1)*Nx*(Nz)};
  cufftPlanMany(&plan,1,n,inembed,(Ny/2+1)*Nx,1,
                          onembed,(Ny/2+1)*Nx,1,CUFFT_C2C,(Ny/2+1)*Nx);
              //    n rank  nembed  stride   dist
	
  zeroC<<<dimGrid,dimBlock>>>(result);
  
  cufftExecC2C(plan, f, result, CUFFT_FORWARD);
  
  printf("A\n");
  
  getfcnZCOMPLEX(result);
  
  //f is a field of the form f(ky,kx,kz)
  
    
  zderiv<<<dimGrid, dimBlock>>> (result, kz);
  //printf("B\n");
  //getfcnZCOMPLEX(result);
  
  mask_Z<<<dimGrid,dimBlock>>>(result);
  
  cufftExecC2C(plan, result, result, CUFFT_INVERSE);				
  
  
  //now we have a field result of form result(ky,kx,z)
  
  scaler = (float)1/(Nz);
  
  scale<<<dimGrid,dimBlock>>> (result, scaler);
  
  //mask<<<dimGrid,dimBlock>>>(result);
  
  
  cufftDestroy(plan);
  
  //cudaFree(a_complex_z_d); cudaFree(b_complex_z_d);
 
}  


 
