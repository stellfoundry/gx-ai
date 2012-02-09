void courant(float* dt, cufftComplex* zpK, cufftComplex* zmK, cufftComplex* zp, cufftComplex* zm,
                       float* kx, float* ky)
{
    cufftComplex *padded;
    cudaMalloc((void**) &padded, sizeof(cufftComplex)*Nx*Ny*Nz);
    
    cufftComplex *max;
    max = (cufftComplex*) malloc(sizeof(cufftComplex));
    //vymax = (cufftComplex*) malloc(sizeof(cufftComplex));
    //vxmax2 = (cufftComplex*) malloc(sizeof(cufftComplex));
    //vymax2 = (cufftComplex*) malloc(sizeof(cufftComplex));
    
    float vxmax, vymax;
    
    int dev;
    struct cudaDeviceProp prop;
    cudaGetDevice(&dev);
    cudaGetDeviceProperties(&prop,dev);
    int zThreads = prop.maxThreadsDim[2];
    int totalThreads = prop.maxThreadsPerBlock;   
    
    int xy = totalThreads/Nz;
    int blockxy = sqrt(xy);
    //dimBlock = threadsPerBlock, dimGrid = numBlocks
    dim3 dimBlock2(blockxy,blockxy,Nz);
    if(Nz>zThreads) {
      dimBlock2.x = sqrt(totalThreads/zThreads);
      dimBlock2.y = sqrt(totalThreads/zThreads);
      dimBlock2.z = zThreads;
    }  
    
    dim3 dimGrid2(Nx/dimBlock2.x+1,Ny/dimBlock2.y+1,1);
    
    ///////////////////////////////////////////////////////
    
    //calculate max(ky*zp)
    
    multKy<<<dimGrid2,dimBlock2>>>(zpK, zp,ky);
    //zpK = ky*zp
    
    maxReduc(max,zpK,padded); 
       
    
    vxmax = sqrt(max[0].x*max[0].x+max[0].y*max[0].y);		        
    
    /////////////////////////////////////////////////////////
    
    //calculate max(ky*zm)
    
    multKy<<<dimGrid2,dimBlock2>>>(zmK,zm,ky);
    //zmK = ky*zm
    
    maxReduc(max,zmK,padded);
    
    if( sqrt(max[0].x*max[0].x+max[0].y*max[0].y) > vxmax) {
      vxmax = sqrt(max[0].x*max[0].x+max[0].y*max[0].y);
    }  
   
    //////////////////////////////////////////////////////////
    
    //calculate max(kx*zp)
    
    multKx<<<dimGrid2,dimBlock2>>>(zpK, zp,kx);
    //zpK = kx*zp
    
    maxReduc(max,zpK,padded);
    
    vymax = sqrt(max[0].x*max[0].x+max[0].y*max[0].y);
    		     
    ///////////////////////////////////////////////////////
    
    //calculate max(kx*zm)
    
    multKx<<<dimGrid2,dimBlock2>>>(zmK,zm,kx);
    //zmK = kx*zm
    
    maxReduc(max,zmK,padded);
    
    if( sqrt(max[0].x*max[0].x+max[0].y*max[0].y) > vymax) {
      vymax = sqrt(max[0].x*max[0].x+max[0].y*max[0].y);
    }  
    
    /////////////////////////////////////////////////////////
    
    //find dt
    
    if( 2*M_PI/(vxmax*Nx) > 2*M_PI/(vymax*Ny) ) {
      dt[0] = 2*M_PI/(vymax*Ny);
    } else {
      dt[0] = 2*M_PI/(vxmax*Nx);
    }  
    
    
    cudaFree(padded);
    
}    
    
    
    		     
