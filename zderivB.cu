void ZDerivB(float* result, float* bmag, cuComplex* bmag_complex, float* kz) 
{
  double scaler;

  cufftExecR2C(ZDerivBplanR2C, bmag, bmag_complex);
    
  zderiv<<<dimGrid, dimBlock>>> (bmag_complex, bmag_complex, kz, 1,1,Nz/2+1);
  
  cufftExecC2R(ZDerivBplanC2R, bmag_complex, result);				
  
  
  //now we have a field result of form result(z)
  scaler = (double) gradpar/Nz;
  
  scaleReal<<<dimGrid,dimBlock>>> (result, result, scaler, 1,1,Nz);
  
  
 
}  
