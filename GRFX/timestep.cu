void timestep(cufftComplex *zpNew, cufftComplex *zpOld, 
              cufftComplex *zmNew, cufftComplex *zmOld,
	      float* kx, float* ky, float* kz, 
	      float* kPerp2, float* kPerp2Inv, float nu, float eta, float dt)
{
    
    cufftComplex *ZDeriv;
    cufftComplex *zK;
    cufftComplex *bracket1, *bracket2, *brackets;
    
    
    cudaMalloc((void**) &ZDeriv, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &zK, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &bracket1, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &bracket2, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &brackets, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
       
    
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
    
    
    //we will pass in initialized k's
    //kInit<<<dimGrid,dimBlock>>> (kx,ky,kz); 
   
    
    //ZDeriv will be recycled, ie don't distinguish between zp/zm/old/star/new, etc
    //same for zK = kPerp2*z
    
    zeroC<<<dimGrid,dimBlock>>>(ZDeriv);
    
    /////////////////////////
    //A+    
    
    ZDERIV(ZDeriv,zpOld,kz);
    //0) ZDeriv= d/dz(zpOld)

    //bracket1, bracket2, and brackets will be recycled
    multKPerp<<<dimGrid,dimBlock>>> (zK,zmOld,kPerp2, 1);
    //1) zK = kPerp2*zmOld
    
    NLPS(bracket1,zpOld,zK,kx,ky);
    //2) bracket1 = {zp,kperp2*zm}
    
    
    multKPerp<<<dimGrid,dimBlock>>> (zK,zpOld,kPerp2,1);
    //3) zK = kPerp2*zpOld
    
    NLPS(bracket2,zmOld,zK,kx,ky);
    //4) bracket2 = {zm,kPerp2*zp}
    
    addsubt<<<dimGrid,dimBlock>>> (bracket1, bracket1, bracket2, 1);  //result put in bracket1
    //5) bracket1 = {zp,kPerp2*zm}+{zm,kPerp2*zp}
    
    damping<<<dimGrid,dimBlock>>> (bracket1, zpOld,zmOld,kPerp2,nu,1);
    //6) bracket1 = {zp,kPerp2*zm}+{zm,kPerp2*zp} - nu*kPerp2*kPerp2*(zpOld+zmOld)
    
    NLPS(bracket2,zpOld,zmOld,kx,ky);
    //7) bracket2 = {zp,zm}
    
    damping<<<dimGrid,dimBlock>>> (bracket2, zpOld,zmOld,kPerp2,eta,-1);
    //8) bracket2 = {zp,zm} + eta*kPerp2*(zpOld-zmOld)
    
    
    multKPerp<<<dimGrid,dimBlock>>> (bracket2, bracket2,kPerp2,1);
    //9) bracket2 = kPerp2*[{zp,zm} + eta*kPerp2*(zpOld-zmOld)]
    
    //bracket1 and bracket2 are same for A+ and A-, only difference is whether they are added
    //or subtracted
    addsubt<<<dimGrid,dimBlock>>> (brackets,bracket1,bracket2, -1); //result put in brackets
    //10) brackets = [{zp,kPerp2*zm}+{zm,kPerp2*zp}-nu*kPerp2*kPerp2*(zpOld+zmOld)] - kPerp2*[{zp,zm}+ eta*kPerp2*(zpOld-zmOld)]
    
    multKPerp<<<dimGrid,dimBlock>>> (brackets,brackets,kPerp2Inv,1);
    //11) brackets = (1/(2*kPerp2))*brackets    
   
    //here, "zpNew" is actually zpStar
    step<<<dimGrid,dimBlock>>> (zpNew,zpOld,ZDeriv,brackets,dt/2,1);
    //12) zpStar = zpOld + (dt/2)*(ZDeriv - brackets)
    
    
    //////////////////
    //A-
    
    ZDERIV(ZDeriv,zmOld,kz);
    //13) ZDeriv = d/dz(zmOld)
    
    
    
    addsubt<<<dimGrid,dimBlock>>> (brackets,bracket1,bracket2, 1);
    //14) brackets = [{zp,kPerp2*zm}+{zm,kPerp2*zp}-nu*kPerp2*(zpOld+zmOld)] + kPerp2*[{zp,zm}+ eta*kPerp2*(zpOld-zmOld)]
    
    multKPerp<<<dimGrid,dimBlock>>> (brackets,brackets,kPerp2Inv,1);
    //15) brackets = (1/(2*kPerp2))*brackets
    
    //again, "zmNew" is actually zmStar
    step<<<dimGrid,dimBlock>>> (zmNew,zmOld,ZDeriv,brackets,dt/2,-1);
    //16) zmStar = zmOld - (dt/2)*(ZDeriv + brackets)
    
    //////////////////
    //B+
  
    
    ZDERIV(ZDeriv,zpNew,kz);
    //ZDeriv = d/dz(zpStar)
    
    multKPerp<<<dimGrid,dimBlock>>> (zK,zmNew,kPerp2,1); 
    //zK = kPerp2*zmStar
       
    NLPS(bracket1,zpNew,zK,kx,ky);
    //bracket1 = {zpStar,kPerp2*zmStar}
    
    multKPerp<<<dimGrid,dimBlock>>> (zK,zpNew,kPerp2,1);
    //zK = kPerp2*zpStar
    
    NLPS(bracket2,zmNew,zK,kx,ky);
    //bracket2 = {zmStar,kPerp2*zpStar}
    
    addsubt<<<dimGrid,dimBlock>>> (bracket1,bracket1,bracket2, 1);
    //bracket1 = {zpStar,kPerp2*zmStar}+{zmStar,kPerp2*zpStar}
    
    damping<<<dimGrid,dimBlock>>> (bracket1, zpNew,zmNew,kPerp2,nu,1);
    //bracket1 = {zp,kPerp2*zm}+{zm,kPerp2*zp} - nu*kPerp2*(zpStar+zmStar)
    
    NLPS(bracket2,zpNew,zmNew,kx,ky);
    //bracket2 = {zpStar,zmStar}
    
    damping<<<dimGrid,dimBlock>>> (bracket2, zpNew,zmNew,kPerp2,eta,-1);
    //bracket2 = {zp,zm} + eta*kPerp2*(zpStar-zmStar)
    
    multKPerp<<<dimGrid,dimBlock>>> (bracket2,bracket2,kPerp2,1);
    //bracket2 = kPerp2*[{zp,zm} + eta*kPerp2*(zpStar-zmStar)]   
    
    addsubt<<<dimGrid,dimBlock>>> (brackets,bracket1,bracket2, -1);
    //brackets = {zpStar,kPerp2*zmStar}+{zmStar,kPerp2*zpStar}-kPerp2*{zpStar,zmStar}
    
    multKPerp<<<dimGrid,dimBlock>>> (brackets,brackets,kPerp2Inv,1);
    //brackets = (1/(2*kPerp2))*({zpStar,kPerp2*zmStar}+{zmStar,kPerp2*zpStar}-kPerp2*{zpStar,zmStar})
    
    //since the "zNew" (which are actually zStar) terms are all encompassed in the brackets term,
    //we can now reuse "zNew" for zNew
    
    
    //now, "zpNew" is really zpNew
    step<<<dimGrid,dimBlock>>> (zpNew,zpOld,ZDeriv,brackets,dt,1);
  
    
    //////////////////////
    //B-
    
    ZDERIV(ZDeriv,zmNew,kz);
    addsubt<<<dimGrid,dimBlock>>> (brackets,bracket1,bracket2, 1);
    multKPerp<<<dimGrid,dimBlock>>> (brackets,brackets,kPerp2Inv,1);
    
    
    //"zmNew" is really zmNew
    step<<<dimGrid,dimBlock>>> (zmNew,zmOld,ZDeriv,brackets,dt,-1);
    
    //now we copy the results, the zNew's, to the zOld's 
    //so that the routine can be called recursively
    
    cudaMemcpy(zpOld, zpNew, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz,
                           cudaMemcpyDeviceToDevice);
    cudaMemcpy(zmOld, zmNew, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz,
                           cudaMemcpyDeviceToDevice);			   
			       
    
    cudaFree(bracket1), cudaFree(bracket2), cudaFree(brackets);
    cudaFree(ZDeriv); cudaFree(zK);

}

