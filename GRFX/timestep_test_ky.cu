void timestep_test_ky(cufftReal* f, cufftReal* g, FILE* ofile)
{
    //host variables
    float *x, *y, *z;
    cufftComplex *totEnergy, *kinEnergy, *magEnergy;
    x = (float*) malloc(sizeof(float)*Nx);
    y = (float*) malloc(sizeof(float)*Ny);
    z = (float*) malloc(sizeof(float)*Nz);
    totEnergy = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny/2+1));
    kinEnergy = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny/2+1));
    magEnergy = (cufftComplex*) malloc(sizeof(cufftComplex)*(Ny/2+1));
    
    
    //device variables
    cufftReal *f_d, *g_d;
    cufftComplex *fC_d, *gC_d;
    cufftComplex *fC1_d, *gC1_d;
    float *kx, *ky, *kz, *kPerp2, *kPerp2Inv;
    float scaler;
    cudaMalloc((void**) &f_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &g_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &fC_d, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &gC_d, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &fC1_d, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &gC1_d, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &kx, sizeof(float)*Nx);
    cudaMalloc((void**) &ky, sizeof(float)*(Ny/2+1));
    cudaMalloc((void**) &kz, sizeof(float)*Nz);
    cudaMalloc((void**) &kPerp2, sizeof(float)*Nx*(Ny/2+1));
    cudaMalloc((void**) &kPerp2Inv, sizeof(float)*Nx*(Ny/2+1));
    cudaMalloc((void**) &scaler, sizeof(float));
    
    int dev;
    struct cudaDeviceProp prop;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop,dev);
    int zThreads = prop.maxThreadsDim[2];
    int totalThreads = prop.maxThreadsPerBlock;   
    
    int xy = totalThreads/Nz;
    int blockxy = sqrt(xy);
    //dimBlock = threadsPerBlock, dimGrid = numBlocks
    dim3 dimBlock(blockxy,blockxy,Nz);
    if(Nz>zThreads) {
      dimBlock.x = sqrt(totalThreads/zThreads);
      dimBlock.y = sqrt(totalThreads/zThreads);
      dimBlock.z = zThreads;
    }  
    
    dim3 dimGrid(Nx/dimBlock.x+1,Ny/dimBlock.y+1,1);    
	
    for(int k=0; k<Nz; k++) {
     for(int j=0; j<Nx; j++) {
      for(int i=0; i<Ny; i++) {
      
      
      y[i] = X0*2*M_PI*(float)(i-Ny/2)/Ny;                             //  
      x[j] = X0*2*M_PI*(float)(j-Nx/2)/Nx;				    //
      z[k] = X0*2*M_PI*(float)(k-Nz/2)/Nz;				    //
      int index = i + Ny*j + Ny*Nx*k;
      
      
      //we use the Orszag-Tang initial conditions
      //phi = -2(cosx + cosy)
      //A = 2cosy + cos2x
      //f = z+ = phi + A
      //g = z- = phi - A
      f[index] = (-cos(x[j]) - 0*cos(y[i]) + .5*cos(2*x[j]));		
      g[index] = (-cos(x[j]) - 2*cos(y[i]) - .5*cos(2*x[j]));
      //g[index] = cos((Nx/2)*x[j]);
      /* f:
         (0,1) -> -2
         (1,0) -> -1
	 (0,2) -> .5
	 g:
	 (0,1) -> -2
	 (1,0) -> -3
	 (0,2) -> -.5	*/	
      	        		   
      }
     }
    } 
    
    
    cudaMemcpy(f_d, f, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d, g, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyHostToDevice);
    
    
    
    cufftHandle plan;
    cufftHandle plan2;
    int n[2] = {Nx, Ny};
    
    cufftPlanMany(&plan, 2,n,NULL,1,0,NULL,1,0,CUFFT_R2C,Nz);
    cufftPlanMany(&plan2,2,n,NULL,1,0,NULL,1,0,CUFFT_C2R,Nz);
    
    cufftExecR2C(plan, f_d, fC_d);
    cufftExecR2C(plan, g_d, gC_d);    
    //now we have fields of the form f(ky,kx,z)
    
    
    
    //IMPORTANT: scale fields immediately after transform to k-space, not after transform back to real space at end of routine
    scaler = (float) 1 / (Nx*Ny/2);
    scale<<<dimGrid,dimBlock>>>(fC_d, scaler);
    scale<<<dimGrid,dimBlock>>>(gC_d, scaler);

    
    roundoff<<<dimGrid,dimBlock>>>(fC_d,.1);
    roundoff<<<dimGrid,dimBlock>>>(gC_d,.1);
    
    printf("f:\n");
    getfcn(fC_d);
    printf("g:\n");
    getfcn(gC_d);
    printf("\n\n");
    
    //return; 
    
    kInit<<<dimGrid, dimBlock>>> (kx,ky,kz);
    kPerpInit<<<dimGrid,dimBlock>>>(kPerp2, kx, ky);
    kPerpInvInit<<<dimGrid,dimBlock>>>(kPerp2Inv,kPerp2);
    
    
    fprintf(ofile, "#\ttime(s)\t\ttotal energy\tkinetic energy\tmagnetic energy\n");
    
    
    float *dt;
    dt = (float*) malloc(sizeof(float));
    float time=0;
    int counter=0;
    
    cudaEvent_t start, stop;
    float runtime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);					    

    cudaEventRecord(start,0);
    
    while(time < endtime) {
      printf("%f      %d\n",time,counter);
      
      courant(dt,fC1_d,gC1_d,fC_d,gC_d,kx,ky);
      //fC_d and gC_d are not modified by courant(); fC1_d and gC1_d are modified
            
      
      //totEnergy, kinEnergy, and magEnergy are now arrays of length (Ny/2+1), so ...Energy[1] is the energy of ky[1]
      energy_ky(totEnergy, kinEnergy, magEnergy, fC1_d,gC1_d, fC_d, gC_d, kPerp2);
      //fC_d and gC_d are not modified by energy(); fC1_d and gC1_d are modified
      
      
      
      //to get different ky's, just change the index of the Energy's 
      fprintf(ofile, "\t%f\t%f\t%f\t%f\n", time/2, totEnergy[0].x/Nz, kinEnergy[0].x/Nz, magEnergy[0].x/Nz);
      
      
      
      timestep(fC1_d,fC_d,gC1_d,gC_d,kx,ky,kz,kPerp2,kPerp2Inv,nu,eta,dt[0]);  
      
      //at end of routine, fC1_d is copied to fC_d, and same for g
      //to allow the routine to be called recursively
      
      printf("%s\n",cudaGetErrorString(cudaGetLastError()));
      
      time+=dt[0];
      counter++;
       
      
    } 
    
    //go one timestep past endtime
    printf("%f      %d\n",time,counter);      
      
    energy_ky(totEnergy, kinEnergy, magEnergy, fC1_d,gC1_d, fC_d, gC_d, kPerp2);
    //fC_d and gC_d are not modified by energy(); fC1_d and gC1_d are modified
      
    fprintf(ofile, "\t%f\t%f\t%f\t%f\n", time/2, totEnergy[0].x/Nz, kinEnergy[0].x/Nz, magEnergy[0].x/Nz);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runtime,start,stop);
    printf("Total time (ms): %f\n",runtime);
    printf("Avg time/timestep (ms): %f\n",runtime/counter);
    
    cufftExecC2R(plan2, fC1_d, f_d);
    cufftExecC2R(plan2, gC1_d, g_d);

    
    scaleReal<<<dimGrid,dimBlock>>>(f_d, .5);
    scaleReal<<<dimGrid,dimBlock>>>(g_d, .5);
    
    
    cudaMemcpy(f, f_d, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
    cudaMemcpy(g, g_d, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
    
    cudaFree(f_d), cudaFree(g_d), cudaFree(fC_d), cudaFree(gC_d);
    cudaFree(fC1_d), cudaFree(gC1_d), cudaFree(kx);
    cudaFree(ky), cudaFree(kz);

}    
    
    
    
    
       	   
