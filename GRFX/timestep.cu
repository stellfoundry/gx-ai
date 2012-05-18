void timestep(cufftComplex *zpNew, cufftComplex *zpOld, 
              cufftComplex *zmNew, cufftComplex *zmOld,
	      float* kx, float* ky, int** kxCover, int** kyCover, int nClasses, int* nLinks, int* nChains, float* kz, 
	      float* kPerp2, float* kPerp2Inv, float nu, float eta, float dt)
{
    
    advance(zpOld,zmOld,zpOld,zmOld,zpNew,zmNew, kx,ky,kz,kxCover,kyCover,nClasses,nLinks,nChains,kPerp2,kPerp2Inv,nu,eta,dt/2);
    
    advance(zpOld,zmOld,zpNew,zmNew,zpOld,zmOld, kx,ky,kz,kxCover,kyCover,nClasses,nLinks,nChains,kPerp2,kPerp2Inv,nu,eta,dt);
    
    //getfcn(zpNew);

}

