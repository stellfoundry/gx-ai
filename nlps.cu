void NLPS(cuComplex *result, cuComplex *f, cuComplex *g, float *kx, float *ky)
{
  //bool NLPSDEBUG = false;

     
  float scaler = (float) 1./(Nx*Ny*2);				
  
  //////////////////////////////////////////
  //////////////////////////////////////////
  // main part of the procedure
  //////////////////////////////////////////
  //////////////////////////////////////////
  
  
  NLPSderivX<<<dimGrid,dimBlock>>>(deriv_nlps,f,kx);
  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR1_nlps);   
  NLPSderivY<<<dimGrid,dimBlock>>>(deriv_nlps,g,ky);
  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR2_nlps);
  multdiv<<<dimGrid,dimBlock>>>(resultR_nlps,derivR1_nlps,derivR2_nlps,1);
  
  NLPSderivY<<<dimGrid,dimBlock>>>(deriv_nlps,f,ky);
  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR1_nlps);
  NLPSderivX<<<dimGrid,dimBlock>>>(deriv_nlps,g,kx);
  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR2_nlps);
  bracket<<<dimGrid,dimBlock>>>(resultR_nlps,resultR_nlps,derivR1_nlps,derivR2_nlps,scaler);  
  
    
  cufftExecR2C(NLPSplanR2C, resultR_nlps, result);  
  
  reality<<<dimGrid,dimBlock>>>(result);
    

  ///////////////////////////////////////////////
  // dealias
  ///////////////////////////////////////////////

  mask <<<dimGrid,dimBlock>>> (result);

  scale<<<dimGrid,dimBlock>>>(result,result,kxfac);

  
    
}

