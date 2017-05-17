void timestep_test(cufftReal* f, cufftReal* g, FILE* ofile)
{
    //host variables
    float totEnergy, kinEnergy, magEnergy;
    float scaler;
    
    totEnergy = kinEnergy = magEnergy = 0;
    
    //device variables
    cufftReal *f_d, *g_d;
    cufftComplex *fC_d, *gC_d;
    cufftComplex *fC1_d, *gC1_d;
    float* tmpX;
    
    if(DEBUG) getError("After declarations");
    
    cudaMalloc((void**) &f_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &g_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &fC_d, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &gC_d, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &fC1_d, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &gC1_d, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &deriv_nlps, sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &derivR1_nlps, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &derivR2_nlps, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &resultR_nlps, sizeof(float)*Nx*Ny*Nz);
    cudaMalloc((void**) &kx, sizeof(float)*Nx);
    cudaMalloc((void**) &tmpX, sizeof(float)*Nx);
    cudaMalloc((void**) &ky, sizeof(float)*(Ny/2+1));
    cudaMalloc((void**) &kz, sizeof(float)*Nz);
    cudaMalloc((void**) &kPerp2, sizeof(float)*Nx*(Ny/2+1));
    cudaMalloc((void**) &kPerp2Inv, sizeof(float)*Nx*(Ny/2+1));
    
    if(DEBUG) getError("After cudaMallocs");
    
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
  
    getNClasses(&nClasses, idxRight, idxLeft, linksR, linksL, n_k, naky, ntheta0, jshift0);
    
    nLinks = (int*) malloc(sizeof(int)*nClasses);
    nChains = (int*) malloc(sizeof(int)*nClasses);
  
    getNLinksChains(nLinks, nChains, n_k, nClasses, naky, ntheta0);
  
    //int **kxCover_h, **kyCover_h;
    //kxCover_h = (int**) malloc(sizeof(int)*nClasses);
    //kyCover_h = (int**) malloc(sizeof(int)*nClasses);
    int *kxCover_h[nClasses];
    int *kyCover_h[nClasses];

    for(int c=0; c<nClasses; c++) {   
      kyCover_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
      kxCover_h[c] = (int*) malloc(sizeof(int)*nLinks[c]*nChains[c]);
    }  
    
    kFill(nClasses, nChains, nLinks, kyCover_h, kxCover_h, linksL, linksR, idxRight, naky, ntheta0); 
  
    
    //these are the device arrays
    int *kxCover[nClasses];
    int *kyCover[nClasses];
    cuComplex *g_covering[nClasses];
    float *kz_covering[nClasses];
    cufftHandle plan_covering[nClasses];
    for(int c=0; c<nClasses; c++) {    
      int n[1] = {nLinks[c]*Nz};
      cufftPlanMany(&plan_covering[c],1,n,NULL,1,0,NULL,1,0,CUFFT_C2C,nChains[c]);
      //kPrint(nLinks[c], nChains[c], kyCover_h[c], kxCover_h[c]); 
      cudaMalloc((void**) &g_covering[c], sizeof(cuComplex)*Nz*nLinks[c]*nChains[c]);
      cudaMalloc((void**) &kz_covering[c], sizeof(float)*Nz*nLinks[c]);
      cudaMalloc((void**) &kxCover[c], sizeof(int)*nLinks[c]*nChains[c]);
      cudaMalloc((void**) &kyCover[c], sizeof(int)*nLinks[c]*nChains[c]);    
      cudaMemcpy(kxCover[c], kxCover_h[c], sizeof(int)*nLinks[c]*nChains[c], cudaMemcpyHostToDevice);
      cudaMemcpy(kyCover[c], kyCover_h[c], sizeof(int)*nLinks[c]*nChains[c], cudaMemcpyHostToDevice);    
    } 
    
    
    if(!QUIET) printf("naky=%d   ntheta0=%d\n\n", naky, ntheta0);
    ///////////////////////////////////////////////////////////////////////////////////////////////////
    
    
    
    cudaMemcpy(f_d, f, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyHostToDevice);
    cudaMemcpy(g_d, g, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyHostToDevice);
    
    if(DEBUG) getError("After memcpy");
    
    
    int NLPSfftdims[2] = {Nx, Ny};
    cufftPlanMany(&NLPSplanR2C, 2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_R2C, Nz);
    cufftPlanMany(&NLPSplanC2R, 2, NLPSfftdims, NULL, 1, 0, NULL, 1, 0, CUFFT_C2R, Nz);
    
    if(DEBUG) getError("After plan");
    
    cufftExecR2C(NLPSplanR2C, f_d, fC_d);
    cufftExecR2C(NLPSplanR2C, g_d, gC_d);    
    //now we have fields of the form f(ky,kx,z)
    if(DEBUG) getError("After fft");
    
    
    
    //IMPORTANT: scale fields immediately after transform to k-space, not after transform back to real space at end of routine
    scaler = (float) 1. / (Nx*Ny);
    scale<<<dimGrid,dimBlock>>>(fC_d, fC_d, scaler);
    scale<<<dimGrid,dimBlock>>>(gC_d, gC_d, scaler);
    scale_ky_neq_0<<<dimGrid,dimBlock>>>(fC_d, 2.);
    scale_ky_neq_0<<<dimGrid,dimBlock>>>(gC_d, 2.);
    
 
    if(!QUIET) {
    printf("f:\n");
    getfcn(fC_d);
    printf("g:\n");
    getfcn(gC_d);
    printf("\n\n");
    }
    
    
    //roundoff<<<dimGrid,dimBlock>>>(fC_d,.001);
    //if(DEBUG) {
      printf("%s\n",cudaGetErrorString(cudaGetLastError()));
      printf("%d %d %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
      printf("%d %d %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
    //}
    //roundoff<<<dimGrid,dimBlock>>>(gC_d,.001);
    
    
    
    //return;
    
    
    kInit<<<dimGrid, dimBlock>>> (kx,ky,kz, tmpX, NO_ZDERIV);
    kPerpInit<<<dimGrid,dimBlock>>>(kPerp2, kx, ky);
    kPerpInvInit<<<dimGrid,dimBlock>>>(kPerp2Inv,kPerp2);
    
    
    fprintf(ofile, "#\ttime(s)\t\ttotal energy\tkinetic energy\tmagnetic energy\n");
    
    
    float dt = 0.01;
    float time=0;
    int counter=0;
    
    cudaEvent_t start, stop;
    float runtime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);					    

    cudaEventRecord(start,0);
    
    //endtime = 0;
    while(time < endtime) {
      if(!QUIET) printf("%f      %d\n",time,counter);
      
      //courant(dt,fC1_d,gC1_d,fC_d,gC_d,kx,ky);
      //fC_d and gC_d are not modified by courant(); fC1_d and gC1_d are modified
      
      //if(dt[0] < .01f) break;      
      
      energy(&totEnergy, &kinEnergy, &magEnergy, fC1_d,gC1_d, fC_d, gC_d, kPerp2);
      //fC_d and gC_d are not modified by energy(); fC1_d and gC1_d are modified
      
      fprintf(ofile, "\t%e\t%e\t%e\t%e\n", time, totEnergy/Nz, kinEnergy/Nz, magEnergy/Nz);
      fflush(NULL);
        
      
      advance(fC_d,gC_d,fC_d,gC_d,fC1_d,gC1_d, kx,ky,kz,kxCover,kyCover,nClasses,nLinks,nChains,kPerp2,kPerp2Inv,nu,eta,dt/2);
      advance(fC_d,gC_d,fC1_d,gC1_d,fC_d,gC_d, kx,ky,kz,kxCover,kyCover,nClasses,nLinks,nChains,kPerp2,kPerp2Inv,nu,eta,dt);
            
      
      //at end of routine, fC1_d is copied to fC_d, and same for g
      //to allow the routine to be called recursively
      
      if(!QUIET) printf("%s\n",cudaGetErrorString(cudaGetLastError()));
      
      time+=dt;
      counter++;
       
      
    } 
    
    //go one timestep past endtime
    if(!QUIET) printf("%f      %d\n",time,counter);      
      
    energy(&totEnergy, &kinEnergy, &magEnergy, fC1_d,gC1_d, fC_d, gC_d, kPerp2);
    //fC_d and gC_d are not modified by energy(); fC1_d and gC1_d are modified
      
    fprintf(ofile, "\t%e\t%e\t%e\t%e\n", time, totEnergy/Nz, kinEnergy/Nz, magEnergy/Nz);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&runtime,start,stop);
    
    printf("\nExecuted test.\nTotal time (ms): %f\n",runtime);
    printf("Avg time/timestep (ms): %f\n",runtime/counter);
    
    
    cufftExecC2R(NLPSplanC2R, fC1_d, f_d);
    cufftExecC2R(NLPSplanC2R, gC1_d, g_d);

    
    scaleReal<<<dimGrid,dimBlock>>>(f_d, f_d, .5);
    scaleReal<<<dimGrid,dimBlock>>>(f_d, g_d, .5);
    
    
    cudaMemcpy(f, f_d, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
    cudaMemcpy(g, g_d, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
    
    cudaFree(f_d), cudaFree(g_d), cudaFree(fC_d), cudaFree(gC_d);
    cudaFree(fC1_d), cudaFree(gC1_d), cudaFree(kx);
    cudaFree(ky), cudaFree(kz); cudaFree(kxCover); cudaFree(kyCover);
    
    cudaFree(deriv_nlps);
    cudaFree(derivR1_nlps);
    cudaFree(derivR2_nlps);
    cudaFree(resultR_nlps);
  
    cufftDestroy(NLPSplanR2C);
    cufftDestroy(NLPSplanC2R);

}    
    
    
    
    
       	   
