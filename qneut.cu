void qneut(cuComplex* Phi, cuComplex* Apar, cuComplex** Dens, cuComplex** Tprp, cuComplex** Upar, cuComplex** Qprp, cuComplex* PhiAvgNum_tmp, 
           cuComplex* nbar_tmp, cuComplex* nbartot_field, specie* species, specie* species_d, bool adiabatic, float fapar, float beta)
{
#ifdef PROFILE
PUSH_RANGE("qneut", 0);
#endif
  //calculate the real-space ion density (summed over species)
  cudaMemset(nbartot_field, 0., sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  cudaMemset(nbar_tmp, 0., sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  if(adiabatic) {
    for(int s=0; s<nSpecies; s++) {
      convert_guiding_center_to_particle_space<<<dimGrid,dimBlock>>>(nbar_tmp, Dens[s], Tprp[s], species[s], kx, ky, shat, gds2, gds21, gds22, bmagInv);
      add_scaled<<<dimGrid,dimBlock>>>(nbartot_field, 1., nbartot_field, 1., nbar_tmp);
    }
    if(iphi00==1) {
      qneutETG<<<dimGrid,dimBlock>>>(Phi, nbartot_field, species, kx, ky, shat, gds2, gds21, gds22, bmagInv, tau);
    }
    if(iphi00==2) {
      qneutAdiab_part1<<<dimGrid,dimBlock>>>(PhiAvgNum_tmp, nbartot_field, jacobian, species_d, 
					     kx, ky, shat, gds2, gds21, gds22, bmagInv, tau);
      qneutAdiab_part2<<<dimGrid,dimBlock>>>(Phi, PhiAvgNum_tmp, nbartot_field, PhiAvgDenom, jacobian, species_d, 
					     kx, ky, shat, gds2, gds21, gds22, bmagInv, tau);
    }
  } else {
    for(int s=0; s<nSpecies-1; s++) { // electrons are last species, so don't include them in this sum of ion densities
      convert_guiding_center_to_particle_space<<<dimGrid,dimBlock>>>(nbar_tmp, Dens[s], Tprp[s], species[s], kx, ky, shat, gds2, gds21, gds22, bmagInv);
      add_scaled<<<dimGrid,dimBlock>>>(nbartot_field, 1., nbartot_field, 1., nbar_tmp);
    }
    qneut<<<dimGrid,dimBlock>>>(Phi, nbartot_field, Dens[nSpecies], species_d,
					     kx, ky, shat, gds2, gds21, gds22, bmagInv, tau);
    if(fapar > 0.) {
      for(int s=0; s<nSpecies-1; s++) { // electrons are last species, so don't include them in this sum of ion velocities
        // for electromagnetic ions, the 'upar' and 'qprp' evolved by the code are actually 
        // 'upar' = upar + vt*zt*apar_u
        // 'qprp' = qprp + vt*zt*apar_flr
        // here we subtract off the apar parts
        // we will add them back later
        phi_u <<<dimGrid, dimBlock>>> (nbar_tmp, Apar, species[s].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        add_scaled<<<dimGrid,dimBlock>>>(Upar[s], 1., Upar[s], -species[s].vt*species[s].zt, nbar_tmp);
        
        phi_flr <<<dimGrid, dimBlock>>> (nbar_tmp, Apar, species[s].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        add_scaled<<<dimGrid,dimBlock>>>(Qprp[s], 1., Qprp[s], -species[s].vt*species[s].zt, nbar_tmp);

        convert_guiding_center_to_particle_space<<<dimGrid,dimBlock>>>(nbar_tmp, Upar[s], Qprp[s], species[s], kx, ky, shat, gds2, gds21, gds22, bmagInv);
        add_scaled<<<dimGrid,dimBlock>>>(nbartot_field, 1., nbartot_field, 1., nbar_tmp);

        // here we add them back 
        phi_u <<<dimGrid, dimBlock>>> (nbar_tmp, Apar, species[s].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        add_scaled<<<dimGrid,dimBlock>>>(Upar[s], 1., Upar[s], species[s].vt*species[s].zt, nbar_tmp);
        
        phi_flr <<<dimGrid, dimBlock>>> (nbar_tmp, Apar, species[s].rho, kx, ky, shat, gds2, gds21, gds22, bmagInv);
        add_scaled<<<dimGrid,dimBlock>>>(Qprp[s], 1., Qprp[s], species[s].vt*species[s].zt, nbar_tmp);
      }
      ampere<<<dimGrid,dimBlock>>>(Apar, nbartot_field, Upar[nSpecies], beta,
					     kx, ky, shat, gds2, gds21, gds22, bmagInv, tau);
    }
  }
#ifdef PROFILE
POP_RANGE;
#endif
}
