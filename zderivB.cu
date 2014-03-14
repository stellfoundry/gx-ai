void ZDerivB(float* result, float* bmag, cuComplex* bmag_complex, float* kz, cuComplex* bmag_complex_h) 
{
  double scaler;
  char filename[80];
  //zeroC<<<dimGrid,dimBlock>>>(bmag_complex,1,1,Nz/2+1); 

  cufftExecR2C(ZDerivBplanR2C, bmag, bmag_complex);
  if(DEBUG) getError("After bgrad R2C FFT"); 
    
  zderiv<<<dimGrid, dimBlock>>> (bmag_complex, bmag_complex, kz, 1,1,Nz/2+1);
  if(DEBUG) getError("After bgrad ikz");
  //if(DEBUG) fieldWrite(bmag_complex, bmag_complex_h, "bmag_complex.field", filename, 1, 1, Nz/2+1);
  
  cufftExecC2R(ZDerivBplanC2R, bmag_complex, result);				
  if(DEBUG) getError("After bgrad C2R FFT");
  
  
  //now we have a field result of form result(z)
  scaler = (double) gradpar/Nz;
  if(DEBUG) printf("scaler = %f\n", scaler); 
  
  scaleReal<<<dimGrid,dimBlock>>> (result, result, scaler, 1,1,Nz);
  
  
 
}  
