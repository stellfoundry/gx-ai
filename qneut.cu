void qneut(cuComplex* Phi, cuComplex** Dens, cuComplex** Tprp, cuComplex* PhiAvgNum_tmp, cuComplex* nbar_tmp, cuComplex* nbartot_field, specie* species, specie* species_d)
{
  //calculate the real-space ion density (summed over species)
  cudaMemset(nbartot_field, 0., sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  cudaMemset(nbar_tmp, 0., sizeof(cuComplex)*Nx*(Ny/2+1)*Nz);
  for(int s=0; s<nSpecies; s++) {
    nbar<<<dimGrid,dimBlock>>>(nbar_tmp, Dens[s], Tprp[s], species[s], kx, ky, shat, gds2, gds21, gds22, bmagInv);
    add_scaled<<<dimGrid,dimBlock>>>(nbartot_field, 1., nbartot_field, 1., nbar_tmp);
  }
  if(iphi00==1) {
    qneutETG<<<dimGrid,dimBlock>>>(Phi, nbartot_field, species, kx, ky, shat, gds2, gds21, gds22, bmagInv, tau);
  }
  if(iphi00==2) {
    qneutAdiab_part1<<<dimGrid,dimBlock>>>(PhiAvgNum_tmp, nbartot_field, jacobian, species_d, kx, ky, shat, gds2, gds21, gds22, bmagInv, tau);
    qneutAdiab_part2<<<dimGrid,dimBlock>>>(Phi, PhiAvgNum_tmp, nbartot_field, PhiAvgDenom, jacobian, species_d, kx, ky, shat, gds2, gds21, gds22, bmagInv, tau);
  }
}
