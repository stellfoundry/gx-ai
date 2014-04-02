void qneut(cuComplex* Phi, cuComplex** Dens, cuComplex** Tprp, cuComplex* PhiAvgNum_tmp, cuComplex* nbar_tmp, cuComplex* nbartot_field, specie* species)
{
  if(iphi00==1) {
    qneutETG<<<dimGrid,dimBlock>>>(Phi, nbartot_field, species, kx, ky, shat, gds2, gds21, gds22, bmagInv, tau);
  }
  if(iphi00==2) {
    //calculate the real-space ion density (summed over species)
    for(int s=0; s<nSpecies; s++) {
      nbar<<<dimGrid,dimBlock>>>(nbar_tmp, Dens[s], Tprp[s], species[s], kx, ky, shat, gds2, gds21, gds22, bmagInv);
      accum<<<dimGrid,dimBlock>>>(nbartot_field, nbar_tmp, 1);
    }
    qneutAdiab<<<dimGrid,dimBlock>>>(Phi, PhiAvgNum_tmp, nbartot_field, PhiAvgDenom, jacobian, species, kx, ky, shat, gds2, gds21, gds22, bmagInv, tau);
  }
}
