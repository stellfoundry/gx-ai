void energy_ky(cufftComplex* totEnergy_h, cufftComplex* kinEnergy_h, cufftComplex* magEnergy_h,
	       cufftComplex* kPhi, cufftComplex* kA, 
	       cufftComplex* zp, cufftComplex* zm, float* kPerp2)
{
  //cufftComplex *padded;
  //cudaMalloc((void**) &padded, sizeof(cufftComplex)*Nx*Ny*Nz);
  cufftComplex *fky;
  cudaMalloc((void**) &fky, sizeof(cufftComplex)*Nx*Nz);
           
  addsubt<<<dimGrid,dimBlock>>> (kPhi, zp, zm, 1);
  //kPhi = zp+zm
        
  scale<<<dimGrid,dimBlock>>> (kPhi, kPhi, .5);
  //kPhi = .5*(zp+zm) = phi       
    
  addsubt<<<dimGrid,dimBlock>>> (kA, zp, zm, -1);
  //kA = zp-zm
    
  scale<<<dimGrid,dimBlock>>> (kA, kA, .5);
  //kA = .5*(zp-zm) = A
    
  //since the R2C FFT duplicates some elements when ky=0 or ky=Ny/2, we have to fix this by zeroing the duplicate elements
  //before integrating
  fixFFT<<<dimGrid,dimBlock>>>(kPhi);
  fixFFT<<<dimGrid,dimBlock>>>(kA);
            
  squareComplex<<<dimGrid,dimBlock>>> (kPhi);
  //kPhi = phi**2
         
  squareComplex<<<dimGrid,dimBlock>>> (kA);
  //kA = A**2
            
  multKPerp<<<dimGrid,dimBlock>>> (kPhi, kPhi, kPerp2,-1);
  //kPhi = (kperp**2) * (phi**2)
            
  multKPerp<<<dimGrid,dimBlock>>> (kA, kA, kPerp2,-1);
  //kA = (kperp**2) * (A**2)
        
  //loop through the ky's
  for(int i=0; i<Ny/2+1; i++) {
    //for each ky, copy a part of the kPhi array to fky, and sum
    kycopy<<<dimGrid,dimBlock>>> (fky, kPhi, i);
    //since the fky array is Nx*Nz, and this is assumed to be power of 2, we don't need the padded array for the reduction
    sumReduc_nopad(&kinEnergy_h[i], fky);

    //similarly for kA
    kycopy<<<dimGrid,dimBlock>>> (fky, kA, i);
    sumReduc_nopad(&magEnergy_h[i], fky);
    totEnergy_h[i].x = kinEnergy_h[i].x + magEnergy_h[i].x;
  }                
  cudaFree(fky);      
}    
     
    
