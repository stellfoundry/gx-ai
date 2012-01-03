void timestep_test(cufftReal* f, cufftReal* g, int fkx, int fky, int fkz, int fsin,int fcos, 
                   int gkx, int gky, int gkz, int gsin, int gcos, FILE* ofile)
{
    //host variables
    float *x, *y, *z;
    cufftComplex *totEnergy, *kinEnergy, *magEnergy;
    x = (float*) malloc(sizeof(float)*Nx);
    y = (float*) malloc(sizeof(float)*Ny);
    z = (float*) malloc(sizeof(float)*Nz);
    totEnergy = (cufftComplex*) malloc(sizeof(cufftComplex));
    kinEnergy = (cufftComplex*) malloc(sizeof(cufftComplex));
    magEnergy = (cufftComplex*) malloc(sizeof(cufftComplex));
    
    
    //device variables
    cufftReal *f_d, *g_d;
    cufftComplex *fC_d, *gC_d;
    cufftComplex *fC1_d, *gC1_d;
    //cufftComplex *padded_d;
    float *kx, *ky, *kz, *kPerp2, *kPerp2Inv;
    float scaler;
    cudaMalloc((void**) &f_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &g_d, sizeof(cufftReal)*Nx*Ny*Nz);
    cudaMalloc((void**) &fC_d, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &gC_d, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &fC1_d, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &gC1_d, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    //cudaMalloc((void**) &padded_d, sizeof(cufftComplex)*Nx*Ny*Nz);
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
      
      
      y[i] = 2*M_PI*(float)(i-Ny/2)/Ny;                             //  
      x[j] = 2*M_PI*(float)(j-Nx/2)/Nx;				    //
      z[k] = 2*M_PI*(float)(k-Nz/2)/Nz;				    //
      int index = i + Ny*j + Ny*Nx*k;
      
      //we start with two fields f and g of the form f(y,x,z)    
      /*f[index]= fcos*cos(fky*y[i] + fkx*x[j] + fkz*z[k]) + 		//
                fsin*sin(fky*y[i] + fkx*x[j] + fkz*z[k]);	        //
      g[index]= gcos*cos(gky*y[i] + gkx*x[j] + gkz*z[k]) + 		//
                gsin*sin(gky*y[i] + gkx*x[j] + gkz*z[k]);  
		*/
      //we use the Orszag-Tang initial conditions
      //phi = -2cosx - 2sinx
      //A = cosy + .5cos2x
      //f = z+ = phi + A
      //g = z- = phi - A
      f[index] = -2*cos(x[j]) - cos(y[i]) + .5*cos(2*x[j]);		
      g[index] = -2*cos(x[j]) - 3*cos(y[i]) - .5*cos(2*x[j]);
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
    
    
    printf("f:\n");
    getfcn(fC_d);
    printf("g:\n");
    getfcn(gC_d);
    //printf("\n\n");
    
    kInit<<<dimGrid, dimBlock>>> (kx,ky,kz);
    kPerpInit<<<dimGrid,dimBlock>>>(kPerp2, kx, ky);
    kPerpInvInit<<<dimGrid,dimBlock>>>(kPerp2Inv,kPerp2);
    
    
    int steps = 200;
    float dt = .01;
    float nu = .5;
    float eta = .5;
   
    fprintf(ofile, "#\ttime(s)\ttotal energy\tkinetic energy\tmagnetic energy\n");
    
    //getfcn(fC_d);
    
    for(int i=0; i<steps; i++) {
      printf("\n%d\n\n",i);
      
      energy(totEnergy, kinEnergy, magEnergy, fC1_d,gC1_d, fC_d, gC_d, kPerp2);
      //fC_d and gC_d are not modified by energy(); fC1_d and gC1_d are modified
      
      fprintf(ofile, "\t%f\t%f\t%f\t%f\n", i*dt, totEnergy[0].x, kinEnergy[0].x, magEnergy[0].x);
      
      timestep(fC1_d,fC_d,gC1_d,gC_d,kx,ky,kz,kPerp2,kPerp2Inv,nu,eta,dt);  
       
      //at end of routine, fC1_d is copied to fC_d, and same for g
      //to allow the routine to be called recursively
    } 
    
    
    //getfcn(fC1_d);
    
    cufftExecC2R(plan2, fC1_d, f_d);
    cufftExecC2R(plan2, gC1_d, g_d);
    
    //scaler = (float) 1/2;
    
    scaleReal<<<dimGrid,dimBlock>>>(f_d, .5);
    scaleReal<<<dimGrid,dimBlock>>>(g_d, .5);
    
    
    cudaMemcpy(f, f_d, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
    cudaMemcpy(g, g_d, sizeof(cufftReal)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
    
    cudaFree(f_d), cudaFree(g_d), cudaFree(fC_d), cudaFree(gC_d);
    cudaFree(fC1_d), cudaFree(gC1_d), cudaFree(kx);
    cudaFree(ky), cudaFree(kz);

}    
    
    
    
    
       	   
