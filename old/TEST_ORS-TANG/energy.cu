void energy(float *totEnergy_h, float *kinEnergy_h, float *magEnergy_h,
              cufftComplex* kPhi, cufftComplex* kA, 
	      cufftComplex* zp, cufftComplex* zm, float* kPerp2)
{
    float* Phi2;
    cudaMalloc((void**) &Phi2, sizeof(float)*Nx*(Ny/2+1)*Nz);    
    
    float scaler = .5;
    
    addsubt<<<dimGrid,dimBlock>>> (kPhi, zp, zm, 1);
    //kPhi = zp+zm
            
    scale<<<dimGrid,dimBlock>>> (kPhi, kPhi, scaler);
    //kPhi = .5*(zp+zm) = phi
  
    
    addsubt<<<dimGrid,dimBlock>>> (kA, zp, zm, -1);
    //kA = zp-zm
    
    scale<<<dimGrid,dimBlock>>> (kA, kA, scaler);
    //kA = .5*(zp-zm) = A
    
    //since the R2C FFT duplicates some elements when ky=0 or ky=Ny/2, we have to fix this by zeroing the duplicate elements
    //before integrating
    //fixFFT<<<dimGrid,dimBlock>>>(kPhi);
    //fixFFT<<<dimGrid,dimBlock>>>(kA);
    
    
    int size = Nx*(Ny/2+1)*Nz;
    
        
    squareComplex<<<dimGrid,dimBlock>>> (Phi2,kPhi);
    //kPhi = phi**2    
    
    fixFFT<<<dimGrid,dimBlock>>>(Phi2);
    
    multKPerp<<<dimGrid,dimBlock>>> (Phi2, Phi2, kPerp2,-1);
    //kPhi = (kperp**2) * (phi**2)
    
    scaleReal<<<dimGrid,dimBlock>>>(Phi2, Phi2, (float) .5, Nx, Ny/2+1, Nz);
    //Phi2 = .5*kperp**2*phi**2
    
    // integrate to find kinetic energy
    *kinEnergy_h = sumReduc(Phi2, size, false);
    
     
    
    squareComplex<<<dimGrid,dimBlock>>> (Phi2,kA);
    //kA = A**2
    
    fixFFT<<<dimGrid,dimBlock>>>(Phi2);
    
    multKPerp<<<dimGrid,dimBlock>>> (Phi2, Phi2, kPerp2,-1);
    //kA = (kperp**2) * (A**2)
            
    scaleReal<<<dimGrid,dimBlock>>>(Phi2, Phi2, (float) .5, Nx, Ny/2+1, Nz);

    // integrate kA to find magnetic energy
    *magEnergy_h = sumReduc(Phi2, size, false);
    
    
    
    //calculate total energy
    *totEnergy_h = *kinEnergy_h + *magEnergy_h;
    
    printf("E= %f     T= %f     U= %f\n", *totEnergy_h, *kinEnergy_h, *magEnergy_h);
    
    
}    
     
    
