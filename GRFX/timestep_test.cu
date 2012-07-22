void timestep_test(cufftReal* f, cufftReal* g, FILE* ofile)
{
    //host variables
    cufftComplex *totEnergy, *kinEnergy, *magEnergy;
    totEnergy = (cufftComplex*) malloc(sizeof(cufftComplex));
    kinEnergy = (cufftComplex*) malloc(sizeof(cufftComplex));
    magEnergy = (cufftComplex*) malloc(sizeof(cufftComplex));
    
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
    
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    //set up kxCover and kyCover for covering space z-transforms
    int naky, ntheta0, jshift0, nshift;
    naky = 1 + (Ny-1)/3;
    ntheta0 = 1 + 2*(Nx-1)/3;     //MASK IN MIDDLE OF ARRAY
    nshift = Nx - ntheta0;
  
    jshift0 = 3;
  
    int idxRight[naky*ntheta0];
    int idxLeft[naky*ntheta0];
  
    int linksR[naky*ntheta0];
    int linksL[naky*ntheta0];
  
    int n_k[naky*ntheta0];
  
    int nClasses;
  
    getNClasses(&nClasses, idxRight, idxLeft, linksR, linksL, n_k, naky, ntheta0, jshift0);
    
    int nLinks[nClasses];
    int nChains[nClasses];
  
    getNLinksChains(nLinks, nChains, n_k, nClasses, naky, ntheta0);
  
    int **kxCover_h, **kyCover_h;
    kxCover_h = (int**) malloc(sizeof(int)*nClasses);
    kyCover_h = (int**) malloc(sizeof(int)*nClasses);
    for(int c=0; c<nClasses; c++) {   
      kyCover_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
      kxCover_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
    }  
  
    kFill(nClasses, nChains, nLinks, kyCover_h, kxCover_h, linksL, linksR, idxRight, naky, ntheta0); 
    
    
    //these are the device arrays
    int *kxCover[nClasses];
    int *kyCover[nClasses];
    for(int c=0; c<nClasses; c++) {
      //kPrint(nLinks[c], nChains[c], kyCover_h[c], kxCover_h[c]);      
      cudaMalloc((void**) &kxCover[c], sizeof(int)*nLinks[c]*nChains[c]);
      cudaMalloc((void**) &kyCover[c], sizeof(int)*nLinks[c]*nChains[c]);
      cudaMemcpy(kxCover[c], kxCover_h[c], sizeof(int)*nLinks[c]*nChains[c], cudaMemcpyHostToDevice);
      cudaMemcpy(kyCover[c], kyCover_h[c], sizeof(int)*nLinks[c]*nChains[c], cudaMemcpyHostToDevice);
    }
    
    
    if(!quiet) printf("naky=%d   ntheta0=%d\n\n", naky, ntheta0);
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    
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

    
    //roundoff<<<dimGrid,dimBlock>>>(fC_d,.001);
    if(debug) {
      printf("%s\n",cudaGetErrorString(cudaGetLastError()));
      printf("%d %d %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
      printf("%d %d %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    }
    //roundoff<<<dimGrid,dimBlock>>>(gC_d,.001);
    
    if(!quiet) {
    printf("f:\n");
    getfcn(fC_d);
    printf("g:\n");
    getfcn(gC_d);
    printf("\n\n");
    }
    
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
    
    //endtime = .01;
    while(time < endtime) {
      if(!quiet) printf("%f      %d\n",time,counter);
      
      courant(dt,fC1_d,gC1_d,fC_d,gC_d,kx,ky);
      //fC_d and gC_d are not modified by courant(); fC1_d and gC1_d are modified
      
      //if(dt[0] < .01f) break;      
      
      energy(totEnergy, kinEnergy, magEnergy, fC1_d,gC1_d, fC_d, gC_d, kPerp2);
      //fC_d and gC_d are not modified by energy(); fC1_d and gC1_d are modified
      
      fprintf(ofile, "\t%f\t%f\t%f\t%f\n", time/2, totEnergy[0].x/Nz, kinEnergy[0].x/Nz, magEnergy[0].x/Nz);
      fflush(NULL);
      
      
      timestep(fC1_d,fC_d,gC1_d,gC_d,kx,ky,kxCover,kyCover,nClasses,nLinks,nChains,kz,kPerp2,kPerp2Inv,nu,eta,dt[0]);  
      
      //at end of routine, fC1_d is copied to fC_d, and same for g
      //to allow the routine to be called recursively
      
      if(!quiet) printf("%s\n",cudaGetErrorString(cudaGetLastError()));
      
      time+=dt[0];
      counter++;
       
      
    } 
    
    //go one timestep past endtime
    if(!quiet) printf("%f      %d\n",time,counter);      
      
    energy(totEnergy, kinEnergy, magEnergy, fC1_d,gC1_d, fC_d, gC_d, kPerp2);
    //fC_d and gC_d are not modified by energy(); fC1_d and gC1_d are modified
      
    fprintf(ofile, "\t%f\t%f\t%f\t%f\n", time/2, totEnergy[0].x/Nz, kinEnergy[0].x/Nz, magEnergy[0].x/Nz);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runtime,start,stop);
    
    printf("\nExecuted test.\nTotal time (ms): %f\n",runtime);
    printf("Avg time/timestep (ms): %f\n",runtime/counter);
    
    
    cufftExecC2R(plan2, fC1_d, f_d);
    cufftExecC2R(plan2, gC1_d, g_d);

    
    scaleReal<<<dimGrid,dimBlock>>>(f_d, .5);
    scaleReal<<<dimGrid,dimBlock>>>(g_d, .5);
    
    
    cudaMemcpy(f, f_d, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
    cudaMemcpy(g, g_d, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
    
    cudaFree(f_d), cudaFree(g_d), cudaFree(fC_d), cudaFree(gC_d);
    cudaFree(fC1_d), cudaFree(gC1_d), cudaFree(kx);
    cudaFree(ky), cudaFree(kz); cudaFree(kxCover); cudaFree(kyCover);
    
    cufftDestroy(plan);
    cufftDestroy(plan2);

}    
    
    
    
    
       	   
