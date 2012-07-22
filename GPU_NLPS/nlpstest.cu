cufftReal* NLPStest(int fkx, int fky, int fkz, int fsin, int fcos, int gkx, int gky, int gkz, int gsin, int gcos) 
{
    //host variables
    cufftReal *f, *g, *nlps;    
    float *y, *x, *z;
    
    f = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    g = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    y = (float*) malloc(sizeof(float)*Ny);                                 //
    x = (float*) malloc(sizeof(float)*Nx);
    z = (float*) malloc(sizeof(float)*Nz);			   //
    nlps = (cufftReal*) malloc(sizeof(cufftReal)*Nx*Ny*Nz);
    
    
    //device variables
    cufftReal *f_d, *g_d;
    cufftComplex *f_complex_d, *g_complex_d;
    
    float scaler;
         
    cudaMalloc((void**) &f_d, sizeof(cufftReal)*Nx*Ny*Nz); 
    cudaMalloc((void**) &g_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &f_complex_d, sizeof(cufftComplex)*((Ny/2+1))*(Nx)*Nz);
    cudaMalloc((void**) &g_complex_d, sizeof(cufftComplex)*((Ny/2+1))*(Nx)*Nz);    
    cudaMalloc((void**) &scaler, sizeof(float));
    

    if(!(fky < Ny && fky >= 0 && fkx > -Nx/2 && fkx < (Nx/2+1) && gky < Ny && gky >= 0 && gkx > -Nx/2 && gkx < (Nx/2+1))) {
      printf("\nWARNING: Aliasing, make sure\n-Nx/2 < kx < Nx/2+1\n0 < ky < Ny/2+1\n\n");
    }
    
    
    for(int k=0; k<Nz; k++) {
     for(int j=0; j<Nx; j++) {
      for(int i=0; i<Ny; i++) {
      
      
      y[i] = 2*M_PI*(float)(i-Ny/2)/Ny;                             
      x[j] = 2*M_PI*(float)(j-Nx/2)/Nx;	
      z[k] = 2*M_PI*(float)(k-Nz/2)/Nz;			    
      int index = i + Ny*j + Ny*Nx*k;
      
           
      f[index]= fcos*cos( fky*y[i] + fkx*x[j] + fkz*z[k]) + fsin*sin(fky*y[i] + fkx*x[j] + fkz*z[k]);	        
      g[index]= gcos*cos( gky*y[i] + gkx*x[j] + gkz*z[k]) + gsin*sin(gky*y[i] + gkx*x[j] + gkz*z[k]);		
      
            
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
    
    cufftExecR2C(plan, f_d, f_complex_d);
    cufftExecR2C(plan, g_d, g_complex_d);
    
    //getfcn(f_complex_d);
    //getfcn(g_complex_d);
    scaler = (float) 1/(Nx*Ny/2);
    scale<<<dimGrid,dimBlock>>>(f_complex_d,scaler);
    if(debug) getError("After first kernel");
    scale<<<dimGrid,dimBlock>>>(g_complex_d,scaler);

    if(debug) {
      printf("\nF:\n"); 
      getfcn(f_complex_d);
      printf("\nG:\n");
      getfcn(g_complex_d);
    }
    
    cudaFree(f_d); cudaFree(g_d);
    
    cufftReal *nlps_d;
    cufftComplex *nlps_complex_d;
    cudaMalloc((void**) &nlps_complex_d, sizeof(cufftComplex)*(Ny/2+1)*Nx*Nz);
    cudaMalloc((void**) &nlps_d, sizeof(cufftReal)*Nx*Ny*Nz);
    

    cudaEvent_t start, stop;
    float time;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);					    

    
    float *kx_d, *ky_d, *kz_d;
    cudaMalloc((void**) &ky_d, sizeof(float)*(Ny/2+1));                                 
    cudaMalloc((void**) &kx_d, sizeof(float)*(Nx));
    cudaMalloc((void**) &kz_d, sizeof(float)*Nz);	
    
     	
    kInit<<<dimGrid, dimBlock>>> (kx_d, ky_d, kz_d);
    			    
        
    zero<<<dimGrid, dimBlock>>> (nlps_d);
    zeroC<<<dimGrid, dimBlock>>> (nlps_complex_d);
    for(int index=0; index<Nx*Ny*Nz; index++) {
      nlps[index] = 0;
    } 
    
    
    cudaEventRecord(start,0);
      
    if(debug) getError("Before NLPS");  
    NLPS(nlps_complex_d, f_complex_d, g_complex_d, kx_d, ky_d);
    if(debug) getError("After NLPS");
    
    /* for(int i=0; i<4; i++) {
    nlps_complex_d= NLPS(f_complex_d, fdxR_d, fdyR_d, g_complex_d, gdxR_d, gdyR_d, kx_d, ky_d, 1);
    } */
    //nlps_complex_d= NLPS(f_complex_d, fdxR_d, fdyR_d, g_complex_d, gdxR_d, gdyR_d, kx_d, ky_d, 0);    
    
    //getfcn(nlps_complex_d);
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time,start,stop);
    printf("NLPS Time (ms): %f\n",time);
    
    cufftExecC2R(plan2, nlps_complex_d, nlps_d);
    
    scaleReal<<<dimGrid,dimBlock>>>(nlps_d, .5);
    if(debug) getError("After last kernel");
    
    cudaFree(nlps_complex_d); cudaFree(f_complex_d); cudaFree(g_complex_d);
    cudaFree(kx_d); cudaFree(ky_d);

    cudaMemcpy(nlps, nlps_d, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
    
    cudaFree(nlps_d);
    
    
    
    return nlps;
}

   
