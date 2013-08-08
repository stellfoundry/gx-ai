void ZDerivB(float* result, float* bmag, cuComplex* bmag_complex, float* kz) 
{
  float scaler;
  

  cufftExecR2C(ZDerivBplanR2C, bmag, bmag_complex);
  
    
  zderivB<<<dimGrid, dimBlock>>> (bmag_complex, bmag_complex, kz);
    
  
  cufftExecC2R(ZDerivBplanC2R, bmag_complex, result);				
  
  
  //now we have a field result of form result(ky,kx,z)
  
  scaler = (float) gradpar/Nz;
  
  scaleRealZ<<<dimGrid,dimBlock>>> (result, result, scaler);
  
  
 
}  
