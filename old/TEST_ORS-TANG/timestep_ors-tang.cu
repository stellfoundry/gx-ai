void advance(cufftComplex *zp, cufftComplex *zm, 
             cufftComplex *zpOld, cufftComplex *zmOld,
	     cufftComplex *zpNew, cufftComplex *zmNew,
	     float* kx, float* ky, float* kz, int** kxCover, int** kyCover, int nClasses, int* nLinks, int* nChains,
	     float* kPerp2, float* kPerp2Inv,
	     float nu, float eta, float dt)
{
    cufftComplex *ZDeriv;
    cufftComplex *zK;
    cufftComplex *bracket1, *bracket2, *brackets;
    
    
    cudaMalloc((void**) &ZDeriv, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &zK, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &bracket1, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &bracket2, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
    cudaMalloc((void**) &brackets, sizeof(cufftComplex)*Nx*(Ny/2+1)*Nz);
     
    
    zeroC<<<dimGrid,dimBlock>>>(ZDeriv);
    
    if(Nz!=1) {
      //ZDerivCovering(ZDeriv,zpOld, kxCover, kyCover, nClasses, nLinks, nChains);
      //ZDERIV(ZDeriv,zpOld, kz);
    }
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
    step<<<dimGrid,dimBlock>>> (zpNew,zp,ZDeriv,brackets,dt,1);
    //12) zpStar = zpOld + (dt/2)*(ZDeriv - brackets)
    
    if(Nz!=1) {
      //ZDerivCovering(ZDeriv,zmOld, kxCover, kyCover, nClasses,nLinks, nChains);
      //ZDERIV(ZDeriv, zmOld, kz);
    }
    //13) ZDeriv = d/dz(zmOld) 
    
    
    addsubt<<<dimGrid,dimBlock>>> (brackets,bracket1,bracket2, 1);
    //14) brackets = [{zp,kPerp2*zm}+{zm,kPerp2*zp}-nu*kPerp2*(zpOld+zmOld)] + kPerp2*[{zp,zm}+ eta*kPerp2*(zpOld-zmOld)]
    
    multKPerp<<<dimGrid,dimBlock>>> (brackets,brackets,kPerp2Inv,1);
    //15) brackets = (1/(2*kPerp2))*brackets
    
    //again, "zmNew" is actually zmStar
    step<<<dimGrid,dimBlock>>> (zmNew,zm,ZDeriv,brackets,dt,-1);
    //16) zmStar = zmOld - (dt/2)*(ZDeriv + brackets)
    
    cudaFree(bracket1), cudaFree(bracket2), cudaFree(brackets);
    cudaFree(ZDeriv); cudaFree(zK);
}    

