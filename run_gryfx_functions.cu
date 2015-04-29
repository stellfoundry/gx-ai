
void calculate_additional_geo_arrays(
    int Nz,
    float * kz,
    float * tmpZ,
		input_parameters_struct * pars, 
		cuda_dimensions_struct * cdims, 
		geometry_coefficents_struct * geo_d, 
		geometry_coefficents_struct * geo_h){

  bmagInit <<< cdims->dimGrid,cdims->dimBlock >>> (geo_d->bmag, geo_d->bmagInv);
  if(pars->igeo==0) 
		jacobianInit <<< cdims->dimGrid, cdims->dimBlock >>> (
			geo_d->jacobian,pars->drhodpsi,geo_d->gradpar,geo_d->bmag
			);
  else {
    //EGH I think this is wrong, should be line below
jacobianInit <<< cdims->dimGrid, cdims->dimBlock >>> (
			geo_d->jacobian,pars->drhodpsi,geo_d->gradpar,geo_d->bmag
			);
    //copy from geo output
    //cudaMemcpy(geo_d->jacobian, geo_h->jacobian, sizeof(float)*grids->Nz, cudaMemcpyHostToDevice);
    
    //calculate bgrad
      //calculate bgrad = d/dz ln(B(z)) = 1/B dB/dz
     if(pars->debug) printf("calculating bgrad\n");
    cudaMemset(geo_d->bmag_complex, 0, sizeof(cuComplex)*(Nz/2+1));
    //NB This function also sets bmag_complex
    ZDerivB(geo_d->bgrad, geo_d->bmag, geo_d->bmag_complex, kz);
    multdiv<<<cdims->dimGrid,cdims->dimBlock>>>(geo_d->bgrad, geo_d->bgrad, geo_d->bmagInv, 1, 1, Nz, 1);
		/*
*/
  }  
    
  
  //for flux calculations
  multdiv <<< cdims->dimGrid,cdims->dimBlock >>> (tmpZ, geo_d->jacobian, geo_d->grho,1,1,Nz,1);
  geo_h->fluxDen = geo_d->fluxDen = sumReduc(tmpZ,Nz,false);
	
	/* Globals: to be deleted eventually */
	fluxDen = geo_d->fluxDen;
  if(DEBUG) getError("run_gryfx.cu, after init"); 
}
