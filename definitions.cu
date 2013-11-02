void definitions()
{
  /*if(gradpar > 1e-8) {
    *&Zp = 1./gradpar;
  }*/
  
  //*&Zp = 1;
  
  for(int s=0; s<nSpecies; s++) {
    species[s].vt = sqrt(species[s].temp/species[s].mass);
    species[s].zstm = species[s].z/sqrt(species[s].temp*species[s].mass);
    species[s].tz = species[s].temp/species[s].z;
    species[s].zt = species[s].z/species[s].temp;
    species[s].rho = sqrt(species[s].temp*species[s].mass)/species[s].z;
  }
  
  
  D_par = 2.*sqrt(M_PI)/(3.0*M_PI-8.0);
  D_prp = sqrt(M_PI)/2.;
  Beta_par = 2.*(32.-9.*M_PI)/(6.*M_PI-16.);
  
  
  Nx_unmasked = 2*Nx/3+1;
  Ny_unmasked = (Ny-1)/3+1;
  
  
  cflx = shat*X0*((float)Nx_unmasked) / ( ((float)(Nx_unmasked/2))*2*M_PI*Y0);
  cfly = ( (float)(2*(Ny_unmasked-1)) ) / (2*M_PI*Y0);
  
  bool default_nu = true;
  
  if(default_nu) {
    nu[1].x=2.019;
    nu[1].y=-1.620;
    nu[2].x=.433;  
    nu[2].y= 1.018;
    nu[3].x=-.256; 
    nu[3].y=1.487; 
    nu[4].x=-.070; 
    nu[4].y=-1.382;
    nu[5].x=-8.927;
    nu[5].y=12.649;
    nu[6].x= 8.094;
    nu[6].y= 12.638;
    nu[7].x= 13.720;
    nu[7].y= 5.139;
    nu[8].x= 3.368; 
    nu[8].y= -8.110;
    nu[9].x= 1.974; 
    nu[9].y= -1.984;
    nu[10].x= 8.269;
    nu[10].y= 2.060;
  }
  
  //varenna
  if(ivarenna == 1) {
  mu[1].x = 0.;
  mu[1].y = 0.; 
  mu[2].x = 0.;
  mu[2].y = 0.; 
  mu[3].x = 0.;
  mu[3].y = 0.;
  mu[4].x = 0.;
  mu[4].y = 0.; 
  mu[5].x = 0.;
  mu[5].y = 0.;
  mu[6].x = 0.;
  mu[6].y = 0.;
  mu[7].x = 0.;
  mu[7].y = 0.;
  mu[8].x = 0.;
  mu[8].y = 0.;
  mu[9].x = 0.;
  mu[9].y = 0.;
  mu[10].x = 0.;
  mu[10].y = 0.;
  }
  if(ivarenna == 2 || ivarenna == 3) {
  mu[1].x = 0.;
  mu[1].y = -3.;
  mu[2].x = 0.;
  mu[2].y = 1.;
  mu[3].x = 0.;
  mu[3].y = 0.;
  mu[4].x = 0.;
  mu[4].y = -1.5;
  mu[5].x = 0.;
  mu[5].y = 0.;
  mu[6].x = 0.;
  mu[6].y = 0.;
  mu[7].x = 0.;
  mu[7].y = 0.;
  mu[8].x = 0.;
  mu[8].y = 0.;
  mu[9].x = 0.;
  mu[9].y = 0.;
  mu[10].x = 0.;
  mu[10].y = 0.;
  }
  

}  
  
