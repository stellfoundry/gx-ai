void energy(float *Energy_h, cuComplex* Phi_tmp, float* Phi2)
{  
  fixFFT<<<dimGrid,dimBlock>>>(Phi_tmp);
    
  // this could be changed to use floats for the results, and then intrinsics from cuComplex.h     
  squareComplex<<<dimGrid,dimBlock>>> (Phi2,Phi_tmp);
  //kPhi = phi**2

  int size = Nx*(Ny/2+1)*Nz;
  
  // integrate kPhi to find kinetic energy
  *Energy_h = sumReduc(Phi2, size, false);
  
}
