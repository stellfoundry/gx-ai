float courant(cuComplex* Phi, cuComplex* KPhi_tmp, cuComplex* Phi_u_tmp, 
              float* RePhi, specie *species)
{
  bool COURANTDEBUG = false;
  
  float dt;  
  float max;
  
  int size = Nx*Ny*Nz;
  
  if(DEBUG && COURANTDEBUG) getError("after courant allocs");
    
  float vxmax = 0;
  float vymax = 0;
  float vmax = 0;
  
  for(int s=0; s<nSpecies; s++) {
    phi_u<<<dimGrid,dimBlock>>>(Phi_u_tmp,Phi,species[s].rho,kx,ky,shat,gds2,gds21,gds22,bmagInv); 
    if(COURANTDEBUG) getError("after courant phi kernel");
        
    ///////////////////////////////////////////////////////

    //calculate max(abs(dPhi_u/dy)) of all species

    NLPSderivY<<<dimGrid,dimBlock>>>(KPhi_tmp, Phi_u_tmp, ky);
    mask<<<dimGrid,dimBlock>>>(KPhi_tmp);
    cufftExecC2R(NLPSplanC2R,KPhi_tmp,RePhi);
    
    abs<<<dimGrid,dimBlock>>> (RePhi, RePhi);    
    scaleReal<<<dimGrid,dimBlock>>>(RePhi,RePhi,.5);  

    max = maxReduc(RePhi,size,false);     

    vxmax = max > vxmax ? max : vxmax;

    ///////////////////////////////////////////////////////  
    //calculate max(abs(dPhi_u/dx)) of all species

    NLPSderivX<<<dimGrid,dimBlock>>>(KPhi_tmp, Phi_u_tmp,kx);
    mask<<<dimGrid,dimBlock>>>(KPhi_tmp);
    cufftExecC2R(NLPSplanC2R,KPhi_tmp,RePhi);
    
    abs<<<dimGrid,dimBlock>>> (RePhi, RePhi);
    scaleReal<<<dimGrid,dimBlock>>>(RePhi,RePhi,.5);  
     
    max = maxReduc(RePhi,size,false);
    
    vymax = max > vymax ? max : vymax;
  }  	
  
  vxmax = vxmax*cflx;
  vymax = vymax*cfly;
  
  vmax = vymax > vxmax ? vymax : vxmax;	   
  
  vmax = vmax > cfl/maxdt ? vmax : cfl/maxdt;  
  
  dt = cfl/vmax;
  	     
    
  /////////////////////////////////////////////////////////    
  //find dt
  
  //printf("vxmax = %e    vymax = %e\n", vxmax, vymax);
    
  /*if( 3*M_PI*cfl*Y0/(vxmax*Nx) > 3*M_PI*cfl*X0/(vymax*Ny) ) {
    dt = 3*M_PI*cfl*X0/(vymax*Ny);        
  } else {
    dt = 3*M_PI*cfl*Y0/(vxmax*Nx);
  }  */
  
  if (dt > maxdt) { dt = maxdt; }
  if(COURANTDEBUG) getError("after dt found");
  //printf("dt = %f\n", dt );


  return dt;
  
  //if(COURANTDEBUG) exit(1);
  
    
}    
    
    
    		     
