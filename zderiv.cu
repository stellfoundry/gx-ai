void ZDeriv(cufftComplex *result, cufftComplex* f, float* kz) 
{
  float scaler;
  
	
  zeroC<<<dimGrid,dimBlock>>>(result);
  
  cufftExecC2C(ZDerivplan, f, result, CUFFT_FORWARD);

  
  //f is a field of the form f(ky,kx,kz)
  
    
  zderiv<<<dimGrid, dimBlock>>> (result, result, kz);
  
  //mask_Z<<<dimGrid,dimBlock>>>(result);
  
  cufftExecC2C(ZDerivplan, result, result, CUFFT_INVERSE);				
  
  
  //now we have a field result of form result(ky,kx,z)
  
  scaler = (float)1/(Nz);
  
  scale<<<dimGrid,dimBlock>>> (result, result, scaler);
  

 
}  


 
