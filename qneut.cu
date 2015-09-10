void qneut(cuComplex* Phi, cuComplex* Apar, cuComplex** Dens, cuComplex** Tprp, cuComplex** Upar, cuComplex** Qprp, cuComplex* PhiAvgNum_tmp, 
           cuComplex* nbar_tmp, cuComplex* nbartot_field, cuComplex* ubar_tmp, cuComplex* ubartot_field, 
           specie* species, specie* species_d, input_parameters_struct* pars) // bool adiabatic, float fapar, float beta, bool snyder_electrons)
{
#ifdef PROFILE
PUSH_RANGE("qneut", 0);
#endif

  bool adiabatic = pars->adiabatic_electrons;
  float fapar = pars->fapar;
  float beta = pars->beta;
  bool snyder_electrons = pars->snyder_electrons;
  bool stationary_ions = pars->stationary_ions;

  //bool stationary_ions = false;
  //bool stationary_ions = true;

  //calculate the real-space ion density (summed over species)
  cudaMemset(nbartot_field, 0., sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  cudaMemset(nbar_tmp, 0., sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  if(adiabatic) {
    for(int s=0; s<nSpecies; s++) {
      convert_guiding_center_to_particle_space<<<dimGrid,dimBlock>>>(nbar_tmp, Dens[s], Tprp[s], species[s], kx, ky, shat, gds2, gds21, gds22, bmagInv);
      add_scaled<<<dimGrid,dimBlock>>>(nbartot_field, 1., nbartot_field, 1., nbar_tmp);
    }
    if(iphi00==1) {
      qneutETG<<<dimGrid,dimBlock>>>(Phi, nbartot_field, species, kx, ky, shat, gds2, gds21, gds22, bmagInv, ti_ov_te);
    }
    if(iphi00==2) {
      qneutAdiab_part1<<<dimGrid,dimBlock>>>(PhiAvgNum_tmp, nbartot_field, jacobian, species_d, 
					     kx, ky, shat, gds2, gds21, gds22, bmagInv, ti_ov_te);
      qneutAdiab_part2<<<dimGrid,dimBlock>>>(Phi, PhiAvgNum_tmp, nbartot_field, PhiAvgDenom, jacobian, species_d, 
					     kx, ky, shat, gds2, gds21, gds22, bmagInv, ti_ov_te);
    }
  } else {
    if(!stationary_ions) {
      for(int s=0; s<nSpecies-1; s++) { // electrons are last species, so don't include them in this sum of ion densities
        convert_guiding_center_to_particle_space<<<dimGrid,dimBlock>>>(nbar_tmp, Dens[s], Tprp[s], species[s], kx, ky, shat, gds2, gds21, gds22, bmagInv);
        add_scaled<<<dimGrid,dimBlock>>>(nbartot_field, 1., nbartot_field, 1., nbar_tmp);
      }
    }
    qneut<<<dimGrid,dimBlock>>>(Phi, nbartot_field, Dens[nSpecies-1], species_d,
					     kx, ky, shat, gds2, gds21, gds22, bmagInv, ti_ov_te);

    cudaMemset(ubartot_field, 0., sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
    if(fapar > 0.) {
          // for electromagnetic ions, the 'upar' and 'qprp' evolved by the code are actually 
          // 'upar' = upar + vt*zt*apar_u
          // 'qprp' = qprp + vt*zt*apar_flr
          // here we subtract off the apar parts
          // we will add them back later
        //for(int s=0; s<nSpecies-1; s++) { // electrons are last species, so don't include them in this sum of ion velocities
        //  phi_u <<<dimGrid, dimBlock>>> (ubar_tmp, Apar, species[s].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        //  add_scaled<<<dimGrid,dimBlock>>>(Upar[s], 1., Upar[s], -species[s].vt*species[s].zt, ubar_tmp);
        //  
        //  phi_flr <<<dimGrid, dimBlock>>> (ubar_tmp, Apar, species[s].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        //  add_scaled<<<dimGrid,dimBlock>>>(Qprp[s], 1., Qprp[s], -species[s].vt*species[s].zt, ubar_tmp);
        //}
      if(!stationary_ions) {
        for(int s=0; s<nSpecies-1; s++) { // electrons are last species, so don't include them in this sum of ion velocities

          convert_guiding_center_to_particle_space<<<dimGrid,dimBlock>>>(ubar_tmp, Upar[s], Qprp[s], species[s], kx, ky, shat, gds2, gds21, gds22, bmagInv);
          add_scaled<<<dimGrid,dimBlock>>>(ubartot_field, 1., ubartot_field, 1., ubar_tmp);
          // ^ this is a running sum for the loop over ion species
          // each ubar_s is weighted by n_s and Z_s

        }
      }

      if(snyder_electrons) {
        solve_ampere_for_upar_e<<<dimGrid,dimBlock>>>(Apar, ubartot_field, Upar[nSpecies-1], beta,
                                                                kx, ky, shat, gds2, gds21, gds22, bmagInv, ti_ov_te, species[nSpecies-1].dens);
      } else {
        solve_ampere_for_apar<<<dimGrid,dimBlock>>>(Apar, ubartot_field, Upar[nSpecies-1], beta,
                                                                kx, ky, shat, gds2, gds21, gds22, bmagInv, ti_ov_te, species[nSpecies-1].dens);
      }

        //for(int s=0; s<nSpecies-1; s++) { // electrons are last species, so don't include them in this sum of ion velocities
        //  // here we add the apar parts back 
        //  phi_u <<<dimGrid, dimBlock>>> (ubar_tmp, Apar, species[s].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        //  add_scaled<<<dimGrid,dimBlock>>>(Upar[s], 1., Upar[s], species[s].vt*species[s].zt, ubar_tmp);
        //  
        //  phi_flr <<<dimGrid, dimBlock>>> (ubar_tmp, Apar, species[s].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        //  add_scaled<<<dimGrid,dimBlock>>>(Qprp[s], 1., Qprp[s], species[s].vt*species[s].zt, ubar_tmp);
        //}
    }
  }
#ifdef PROFILE
POP_RANGE;
#endif
}
