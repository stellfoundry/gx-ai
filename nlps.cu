inline void NLPS(cuComplex *result, cuComplex *phi, cuComplex *moment, float *kx, float *ky)
{
  //bool NLPSDEBUG = false;

     
  float scaler = (float) 1./(Nx*Ny);
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
    max = maxReduc(derivR2_nlps,size,derivR1_nlps,derivR2_nlps);                
    //max = maxReduc(derivR2_nlps,size,false);                
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
    max = maxReduc(derivR2_nlps,size,derivR1_nlps, derivR2_nlps);                
    //max = maxReduc(derivR2_nlps,size,false);                
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
#ifndef GS2_zonal
    if(dt_cfl>maxdt) dt_cfl = maxdt;
#endif
  }
#endif
    
}

inline void NLPM_NLPS(cuComplex *result, cuComplex *phi, cuComplex *moment, float *kx, float *ky)
{
  //bool NLPSDEBUG = false;

     
  float scaler = (float) 1./(Nx*Ny);
  
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
  //abs<<<dimGrid,dimBlock>>>(derivR1_nlps, derivR1_nlps);

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

inline void NLPS_abs(cuComplex *result, cuComplex *phi, cuComplex *moment, float* kx, float *ky, bool hammett_nlpm_interference, bool nlpm_abs_sgn)
{
     
  float scaler = (float) 1./(Nx*Ny);
  float max = 0.;
  float vmax = 0.;
  int size = Nx*Ny*Nz;

if(nlpm_abs_sgn) {
  NLPS(result,phi,moment,kx,ky);
  abs_sgn<<<dimGrid,dimBlock>>>(result, result, moment);
} else {  
  //////////////////////////////////////////
  //////////////////////////////////////////
  // main part of the procedure
  //////////////////////////////////////////
  //////////////////////////////////////////
  
  
  NLPSderivX<<<dimGrid,dimBlock>>>(deriv_nlps,phi,kx);
  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR1_nlps);   
  // derivR1_nlps = vy

  NLPSderivY<<<dimGrid,dimBlock>>>(deriv_nlps,phi,ky);
  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR2_nlps);   
  // derivR2_nlps = vx

  NLPSderivY_abs<<<dimGrid,dimBlock>>>(deriv_nlps,moment,ky);
  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR3_nlps);
  // derivR3_nlps = abs(d/dy)M
  mult_abs1_2<<<dimGrid,dimBlock>>>(resultR_nlps, derivR1_nlps, derivR3_nlps);
  // resultR = abs(vy) abs(d/dy) M

  NLPSderivX_abs<<<dimGrid,dimBlock>>>(deriv_nlps,moment,kx);
  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR3_nlps);
  // derivR3_nlps = abs(d/dx)M
  mult_abs1_2<<<dimGrid,dimBlock>>>(derivR3_nlps, derivR2_nlps, derivR3_nlps);
  // derivR3_nlps = abs(vx) abs(d/dx) M
  add_scaled<<<dimGrid,dimBlock>>>(resultR_nlps, 1., resultR_nlps, 1., derivR3_nlps);
  // resultR = abs(vy) abs(d/dy) M + abs(vx) abs(d/dx) M

  if(hammett_nlpm_interference) {
    // extra 'interference' terms

    NLPSderiv_isgnX_derivY<<<dimGrid,dimBlock>>>(deriv_nlps,moment,kx,ky);
    mask<<<dimGrid,dimBlock>>>(deriv_nlps);
    reality<<<dimGrid,dimBlock>>>(deriv_nlps);
    scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
    cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR3_nlps);
    // derivR3_nlps = -i*sgn(kx)dM/dy
    mult_1_sgn2_3<<<dimGrid,dimBlock>>>(derivR3_nlps, derivR1_nlps, derivR2_nlps, derivR3_nlps);
    // derivR3_nlps = -vy*sgn(vx)*(-i*sgn(kx)dM/dy)
    // minus sign because vx = - d/dy Phi
    add_scaled<<<dimGrid,dimBlock>>>(resultR_nlps, 1., resultR_nlps, -1., derivR3_nlps);
    // resultR = abs(vy) abs(d/dy) M + abs(vx) abs(d/dx) M + vy*sgn(vx)*(-i*sgn(kx)dM/dy)

    NLPSderiv_isgnY_derivX<<<dimGrid,dimBlock>>>(deriv_nlps,moment,kx,ky);
    mask<<<dimGrid,dimBlock>>>(deriv_nlps);
    reality<<<dimGrid,dimBlock>>>(deriv_nlps);
    scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
    cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR3_nlps);
    // derivR3_nlps = -i*sgn(ky)dM/dx
    mult_1_sgn2_3<<<dimGrid,dimBlock>>>(derivR3_nlps, derivR2_nlps, derivR1_nlps, derivR3_nlps);
    // derivR3_nlps = -vx*sgn(vy)*(-i*sgn(ky)dM/dx)
    // minus sign because vx = - d/dy Phi
    add_scaled<<<dimGrid,dimBlock>>>(resultR_nlps, 1., resultR_nlps, -1., derivR3_nlps);
    // resultR = abs(vy) abs(d/dy) M + abs(vx) abs(d/dx) M + vy*sgn(vx)*(-i*sgn(kx)dM/dy) + vx*sgn(vy)*(-i*sgn(ky)dM/dx)
  }
  
  cufftExecR2C(NLPSplanR2C, resultR_nlps, result);  
  
  reality<<<dimGrid,dimBlock>>>(result);
  scale_ky_neq_0<<<dimGrid,dimBlock>>>(result,2.);
    

  ///////////////////////////////////////////////
  // dealias
  ///////////////////////////////////////////////

  mask <<<dimGrid,dimBlock>>> (result);

  scale<<<dimGrid,dimBlock>>>(result,result,kxfac*scaler);
}
    
}

//inline void NLPS_abs(cuComplex *result, cuComplex *phi, cuComplex *moment, float* kx, float *ky, bool hammett_nlpm_interference)
//{
//     
//  float scaler = (float) 1./(Nx*Ny);
//  float max = 0.;
//  float vmax = 0.;
//  int size = Nx*Ny*Nz;
 

//  NLPSderivX<<<dimGrid,dimBlock>>>(deriv_nlps,phi,kx);
//  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
//  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
//  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
//  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR1_nlps);   
//
//  NLPSderivY<<<dimGrid,dimBlock>>>(deriv_nlps,moment,ky);
//  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
//  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
//  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
//  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR2_nlps);
//  multdiv<<<dimGrid,dimBlock>>>(resultR_nlps,derivR1_nlps,derivR2_nlps,1);
//  
//  NLPSderivY<<<dimGrid,dimBlock>>>(deriv_nlps,phi,ky);
//  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
//  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
//  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
//  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR1_nlps);
//
//  NLPSderivX<<<dimGrid,dimBlock>>>(deriv_nlps,moment,kx);
//  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
//  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
//  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
//  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR2_nlps);
//  bracket<<<dimGrid,dimBlock>>>(resultR_nlps,resultR_nlps,derivR1_nlps,derivR2_nlps,scaler);  
//  
//  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,moment,.5);
//  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
//  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
//  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR1_nlps);
//  abs_sgn<<<dimGrid,dimBlock>>>(resultR_nlps, resultR_nlps, derivR1_nlps);
//    
//  cufftExecR2C(NLPSplanR2C, resultR_nlps, result);  
//  
//  reality<<<dimGrid,dimBlock>>>(result);
//  scale_ky_neq_0<<<dimGrid,dimBlock>>>(result,2.);
//    
//
//  ///////////////////////////////////////////////
//  // dealias
//  ///////////////////////////////////////////////
//
//  mask <<<dimGrid,dimBlock>>> (result);
//
//  scale<<<dimGrid,dimBlock>>>(result,result,kxfac);
  
//  //////////////////////////////////////////
//  //////////////////////////////////////////
//  // main part of the procedure
//  //////////////////////////////////////////
//  //////////////////////////////////////////
//  
//  
//  NLPSderivX<<<dimGrid,dimBlock>>>(deriv_nlps,phi,kx);
//  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
//  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
//  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
//  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR1_nlps);   
//  // derivR1_nlps = vy
//
//  NLPSderivY<<<dimGrid,dimBlock>>>(deriv_nlps,phi,ky);
//  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
//  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
//  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
//  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR2_nlps);   
//  // derivR2_nlps = -vx
//
//  NLPSderivX<<<dimGrid,dimBlock>>>(deriv_nlps,moment,kx);
//  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
//  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
//  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
//  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR3_nlps);   
//  // derivR3_nlps = dM/dx
//
//  NLPSderivY<<<dimGrid,dimBlock>>>(deriv_nlps,moment,ky);
//  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
//  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
//  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
//  cufftExecC2R(NLPSplanC2R,deriv_nlps,resultR_nlps);   
//  // resultR_nlps = dM/dy
//
//  interference_weight<<<dimGrid,dimBlock>>>(resultR_nlps, derivR1_nlps, derivR2_nlps, derivR3_nlps, resultR_nlps);
//  // resultR_nlps = W
//
//  NLPSderivY_abs<<<dimGrid,dimBlock>>>(deriv_nlps,moment,ky);
//  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
//  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
//  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
//  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR3_nlps);
//  // derivR3_nlps = abs(d/dy)M
//  mult_abs1_2<<<dimGrid,dimBlock>>>(derivR3_nlps, derivR1_nlps, derivR3_nlps);
//  // derivR3_nlps = abs(vy) abs(d/dy) M
//
//  NLPSderivX_abs<<<dimGrid,dimBlock>>>(deriv_nlps,moment,kx);
//  mask<<<dimGrid,dimBlock>>>(deriv_nlps);
//  reality<<<dimGrid,dimBlock>>>(deriv_nlps);
//  scale_ky_neq_0<<<dimGrid,dimBlock>>>(deriv_nlps,.5);
//  cufftExecC2R(NLPSplanC2R,deriv_nlps,derivR1_nlps);
//  // derivR1_nlps = abs(d/dx)M
//  mult_abs1_2<<<dimGrid,dimBlock>>>(derivR1_nlps, derivR2_nlps, derivR1_nlps);
//  // derivR1_nlps = abs(vx) abs(d/dx) M
//  add_scaled<<<dimGrid,dimBlock>>>(derivR3_nlps, 1., derivR3_nlps, 1., derivR1_nlps);
//  // derivR3_nlps = abs(vy) abs(d/dy) M + abs(vx) abs(d/dx) M
//
//  multdiv<<<dimGrid,dimBlock>>>(resultR_nlps, resultR_nlps, derivR3_nlps, 1);
//  // resultR_nlps = W*[abs(vy) abs(d/dy) M + abs(vx) abs(d/dx) M]
//  
//  cufftExecR2C(NLPSplanR2C, resultR_nlps, result);  
//  
//  reality<<<dimGrid,dimBlock>>>(result);
//  scale_ky_neq_0<<<dimGrid,dimBlock>>>(result,2.);
//    
//
//  ///////////////////////////////////////////////
//  // dealias
//  ///////////////////////////////////////////////
//
//  mask <<<dimGrid,dimBlock>>> (result);
//
//  scale<<<dimGrid,dimBlock>>>(result,result,kxfac*scaler);

    
//}
