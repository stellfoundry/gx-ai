void courant(float* dt, cufftComplex* zpK, cufftComplex* zmK, cufftComplex* zp, cufftComplex* zm,
                       float* kx, float* ky)
{
    cufftComplex *padded;
    cudaMalloc((void**) &padded, sizeof(cufftComplex)*Nx*Ny*Nz);
    
    cufftComplex *max;
    max = (cufftComplex*) malloc(sizeof(cufftComplex));
    
    
    float vxmax, vymax;
    
    
    
    ///////////////////////////////////////////////////////
    
    //calculate max(ky*zp)
    
    multKy<<<dimGrid,dimBlock>>>(zpK, zp,ky);
    //zpK = ky*zp
    
    maxReduc(max,zpK,padded); 
       
    
    vxmax = sqrt(max[0].x*max[0].x+max[0].y*max[0].y);		        
    
    /////////////////////////////////////////////////////////
    
    //calculate max(ky*zm)
    
    multKy<<<dimGrid,dimBlock>>>(zmK,zm,ky);
    //zmK = ky*zm
    
    maxReduc(max,zmK,padded);
    
    if( sqrt(max[0].x*max[0].x+max[0].y*max[0].y) > vxmax) {
      vxmax = sqrt(max[0].x*max[0].x+max[0].y*max[0].y);
    }  
   
    //////////////////////////////////////////////////////////
    
    //calculate max(kx*zp)
    
    multKx<<<dimGrid,dimBlock>>>(zpK, zp,kx);
    //zpK = kx*zp
    
    maxReduc(max,zpK,padded);
    
    vymax = sqrt(max[0].x*max[0].x+max[0].y*max[0].y);
    		     
    ///////////////////////////////////////////////////////
    
    //calculate max(kx*zm)
    
    multKx<<<dimGrid,dimBlock>>>(zmK,zm,kx);
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
    
    
    		     
