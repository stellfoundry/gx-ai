void definitions(everything_struct * ev)
{


  // This is a local reference to species, not the global variable
  specie * species = ev->pars.species;
  // Local for convencience
  int Ny = ev->grids.Ny;
  int Nx = ev->grids.Nx;

  
  for(int s=0; s<ev->pars.nspec; s++) {
    species[s].vt = sqrt(species[s].temp/species[s].mass);
    //species[s].zstm = species[s].z/sqrt(species[s].temp*species[s].mass);
    //species[s].tz = species[s].temp/species[s].z;
    species[s].zt = species[s].z/species[s].temp;
    species[s].rho = sqrt(species[s].temp*species[s].mass)/species[s].z;
  }
  
  if(!ev->pars.adiabatic_electrons) {
    ev->pars.ti_ov_te = species[0].temp / species[ev->pars.nspec-1].temp;
    ev->pars.me_ov_mi = species[ev->pars.nspec-1].mass / species[0].mass;
  }
  
  ev->damps.D_par = 2.*sqrt(M_PI)/(3.0*M_PI-8.0);
  ev->damps.D_prp = sqrt(M_PI)/2.;
  ev->damps.Beta_par = 2.*(32.-9.*M_PI)/(6.*M_PI-16.);
  
  if(ev->pars.no_landau_damping) {
    ev->damps.D_par = 0.;
    ev->damps.D_prp = 0.;
    ev->damps.Beta_par = 0.;
  }
  
  ev->grids.Nx_unmasked = 2*Nx/3+1;
  ev->grids.Ny_unmasked = (Ny-1)/3+1;
  
  int ikx_max = (ev->grids.Nx_unmasked+1)/2;
  int iky_max = ev->grids.Ny_unmasked;
  
  ev->time.cflx = ((float) ikx_max)/ev->pars.x0/ev->pars.cfl;// shat*X0*((float)Nx_unmasked) / ( ((float)(Nx_unmasked/2))*2*M_PI*Y0);
  ev->time.cfly = ((float) iky_max)/ev->pars.y0/ev->pars.cfl; //( (float)(2*(Ny_unmasked-1)) ) / (2*M_PI*Y0);
  
  bool default_nu = true;
  
  // These are local variables for convenience, not the 
  // original globals.
  cuComplex * nu = ev->damps.nu;
  cuComplex * mu = ev->damps.mu;

    nu[1].x=0.;
    nu[1].y=0.;
    nu[2].x=0.;
    nu[2].y=0.;
    nu[3].x=0.;
    nu[3].y=0.;
    nu[4].x=0.;
    nu[4].y=0.;
    nu[5].x=0.;
    nu[5].y=0.;
    nu[6].x=0.;
    nu[6].y=0.;
    nu[7].x=0.;
    nu[7].y=0.;
    nu[8].x=0.;
    nu[8].y=0.;
    nu[9].x=0.;
    nu[9].y=0.;
    nu[10].x=0.;
    nu[10].y=0.;

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

  int ivarenna = ev->pars.ivarenna;
  printf("\n\n\nivarenna is %d\nBLAH\nBLAH\n", ivarenna);
  
  //varenna
  if(abs(ivarenna) == 1 || abs(ivarenna)==4 || ev->pars.no_landau_damping) {
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
  if(abs(ivarenna) == 2 || abs(ivarenna) == 3) {
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


  if(abs(ivarenna) == 5) {
  mu[1].x = 0.;
  mu[1].y = 0.5;
  mu[2].x = 0.;
  mu[2].y = 0.;
  mu[3].x = 0.;
  mu[3].y = 1.;
  mu[4].x = 0.;
  mu[4].y = 0.;
  mu[5].x = 0.;
  mu[5].y = 0.;//2.;
  mu[6].x = 0.;
  mu[6].y = 0.;//8.;
  mu[7].x = 0.;
  mu[7].y = 0.;//1.;
  mu[8].x = 0.;
  mu[8].y = 0.;//2.;
  mu[9].x = 0.;
  mu[9].y = 0.;//2.;
  mu[10].x = 0.;
  mu[10].y = 0.;//5.;
  }
  
  if(abs(ivarenna) == 7 || abs(ivarenna)==6) {
  mu[1].x = 0.;
  mu[1].y = -2.;
  mu[2].x = 0.;
  mu[2].y = 0.;
  mu[3].x = 0.;
  mu[3].y = 0.;
  mu[4].x = 0.;
  mu[4].y = -1.5;
  mu[5].x = 0.;
  mu[5].y = 2.;
  mu[6].x = 0.;
  mu[6].y = 8.;
  mu[7].x = 0.;
  mu[7].y = 1.;
  mu[8].x = 0.;
  mu[8].y = 2.;
  mu[9].x = 0.;
  mu[9].y = 2.;
  mu[10].x = 0.;
  mu[10].y = 5.;
  }

}  
  
