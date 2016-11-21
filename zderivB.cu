void ZDerivB(float* result, float* bmag, cuComplex* bmag_complex, cuComplex* dz_bmag_complex, float* kz) 
{
  double scaler;

  cufftExecR2C(ZDerivBplanR2C, bmag, bmag_complex);

  // NRM: for some reason, in certain situations CUDA is having trouble when the same
  // array is both the input and output of a kernel. This was causing problems in the zderivb kernel
  // when the grid size was large enough... 
  // originally, bmag_complex was passed in as both the input and output of the kernel, as indicated
  // in the commented out line below. this kernel just multiplies by i*kz, but it was resulting in
  // random numbers being placed in bmag_complex. allocating a separate output array, dz_bmag_complex,
  // seems to fix the problem. 
  // NB: NEED TO WATCH OUT FOR THIS BEHAVIOR ELSEWHERE. SEEMS TO BE AT LEAST IN PART DEPENDENT ON RESOLUTION.
  //zderivB<<<dimGrid, dimBlock>>> (bmag_complex, bmag_complex, kz);
  zderivB<<<dimGrid, dimBlock>>> (dz_bmag_complex, bmag_complex, kz);

  cufftExecC2R(ZDerivBplanC2R, dz_bmag_complex, result);				
  
  
  //now we have a field result of form result(z)
  scaler = (double) gradpar/Nz;
  
  scaleReal<<<dimGrid,dimBlock>>> (result, result, scaler, 1,1,Nz);
  
  
 
}  
