inline void NLPS(cuComplex *result, cuComplex *phi, cuComplex *moment, float *kx, float *ky)
{
  //bool NLPSDEBUG = false;

     
  float scaler = (float) 1./(Nx*Ny);	//THIS USED TO HAVE A FACTOR OF 2 			
  float max = 0.;
  float vmax = 0.;
  //float vxmax = 0.;
  //float vymax = 0.;
  int size = Nx*Ny*Nz;

  //printf("factor of 2\n");
  
  //////////////////////////////////////////
  //////////////////////////////////////////
  // main part of the procedure
  //////////////////////////////////////////
  //////////////////////////////////////////
  
  //reality<<<dimGrid,dimBlock>>>(f);
  //reality<<<dimGrid,dimBlock>>>(g);
  
  NLPSderivX<<<dimGrid,dimBlock>>>(deriv_nlps,phi,kx);
  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR1_nlps);   
#ifndef ORSTANG
  if(cfl_flag) {
    abs<<<dimGrid,dimBlock>>> (derivR2_nlps, derivR1_nlps);        
    scaleReal<<<dimGrid,dimBlock>>>(derivR2_nlps,derivR2_nlps,.5*cfly);      
    max = maxReduc(derivR2_nlps,size,false);                
    vmax = max > vmax ? max : vmax;
  }
#endif
  NLPSderivY<<<dimGrid,dimBlock>>>(deriv_nlps,moment,ky);
  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR2_nlps);
  multdiv<<<dimGrid,dimBlock>>>(resultR_nlps,derivR1_nlps,derivR2_nlps,1);
  
  NLPSderivY<<<dimGrid,dimBlock>>>(deriv_nlps,phi,ky);
  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR1_nlps);
#ifndef ORSTANG
  if(cfl_flag) {
    abs<<<dimGrid,dimBlock>>> (derivR2_nlps, derivR1_nlps);        
    scaleReal<<<dimGrid,dimBlock>>>(derivR2_nlps,derivR2_nlps,.5*cflx);                                                      
    max = maxReduc(derivR2_nlps,size,false);                
    vmax = max > vmax ? max : vmax;
  }
#endif
  NLPSderivX<<<dimGrid,dimBlock>>>(deriv_nlps,moment,kx);
  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR2_nlps);
  bracket<<<dimGrid,dimBlock>>>(resultR_nlps,resultR_nlps,derivR1_nlps,derivR2_nlps,scaler);  
  
    
  cufftExecR2C(NLPSplanR2C, resultR_nlps, result);  
  
  reality<<<dimGrid,dimBlock>>>(result);
  scale_ky_neq_0<<<dimGrid,dimBlock>>>(result,2.);
    

  ///////////////////////////////////////////////
  // dealias
  ///////////////////////////////////////////////

  mask <<<dimGrid,dimBlock>>> (result);

  scale<<<dimGrid,dimBlock>>>(result,result,kxfac);

#ifndef ORSTANG
  if(cfl_flag) {
    dt_cfl = 1./vmax; 
#ifdef GS2_zonal
    set_gs2_dt_cfl(dt_cfl * sqrt(2.)); //pass dt_cfl to GS2, with appropriate normalization; this is only on proc 0!
#endif
    if(dt_cfl>maxdt) dt_cfl = maxdt;
  }
#endif
    
}

inline void NLPM_NLPS(cuComplex *result, cuComplex *phi, cuComplex *moment, float *kx, float *ky)
{
  //bool NLPSDEBUG = false;

     
  float scaler = (float) 1./(Nx*Ny);	//THIS USED TO HAVE A FACTOR OF 2 			
  //float max = 0.;
  //float vmax = 0.;
  //float vxmax = 0.;
  //float vymax = 0.;
  //int size = Nx*Ny*Nz;

  //printf("factor of 2\n");
  
  //////////////////////////////////////////
  //////////////////////////////////////////
  // main part of the procedure
  //////////////////////////////////////////
  //////////////////////////////////////////
  
  //reality<<<dimGrid,dimBlock>>>(f);
  //reality<<<dimGrid,dimBlock>>>(g);
  
  NLPSderivX<<<dimGrid,dimBlock>>>(deriv_nlps,phi,kx);
  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR1_nlps);   

  NLPSderivY<<<dimGrid,dimBlock>>>(deriv_nlps,moment,ky);
  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR2_nlps);
  multdiv<<<dimGrid,dimBlock>>>(resultR_nlps,derivR1_nlps,derivR2_nlps,1);  
  
    
  cufftExecR2C(NLPSplanR2C, resultR_nlps, result);  
  
  reality<<<dimGrid,dimBlock>>>(result);
  scale_ky_neq_0<<<dimGrid,dimBlock>>>(result,2.);
    

  ///////////////////////////////////////////////
  // dealias
  ///////////////////////////////////////////////

  mask <<<dimGrid,dimBlock>>> (result);

  scale<<<dimGrid,dimBlock>>>(result,result,kxfac*scaler);

    
}
