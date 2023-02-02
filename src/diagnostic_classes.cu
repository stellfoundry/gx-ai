#include "diagnostic_classes.h"

// base class methods
// add a particular type of spectra to the calculation list
void SpectraDiagnostic::add_spectra(SpectraCalc *spectra)
{
  spectraList.push_back(spectra);
  int varid = spectra->define_nc_variable(varname, nc_group);
  spectraIds.push_back(varid);
}

// write all spectra
void SpectraDiagnostic::write_spectra(float* data)
{
  for(int i=0; i<spectraList.size(); i++) {
    spectraList[i]->write(data, spectraIds[i], ncdf_->nc_grids->time_index, nc_group, isMoments, skipWrite);
  }
}

// set kernel launch dimensions for diagnostic calculation kernels
void SpectraDiagnostic::set_kernel_dims()
{
  if(isMoments) {
    int nyx =  grids_->Nyc * grids_->Nx;
    int nslm = grids_->Nmoms * grids_->Nspecies;

    int nt1 = 16;
    int nb1 = 1 + (nyx-1)/nt1;

    int nt2 = 16;
    int nb2 = 1 + (grids_->Nz-1)/nt2;
    
    dB = dim3(nt1, nt2, 1);
    dG = dim3(nb1, nb2, nslm);
  } else {
    dB = dim3(min(8, grids_->Nyc), min(8, grids_->Nx), min(8, grids_->Nz));
    dG = dim3(1 + (grids_->Nyc-1)/dB.x, 1 + (grids_->Nx-1)/dB.y, 1 + (grids_->Nz-1)/dB.z);  
  }
}

// |Phi|**2 diagnostic class
Phi2Diagnostic::Phi2Diagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
{
  nc_type = NC_FLOAT;
  pars_ = pars;
  grids_ = grids;
  geo_ = geo;
  ncdf_ = ncdf;
  varname = "Phi2";
  isMoments = false;
  set_kernel_dims();
  nc_group = ncdf_->nc_diagnostics->spectra;

  add_spectra(allSpectra->t_spectra);
  add_spectra(allSpectra->kxt_spectra);
  add_spectra(allSpectra->kyt_spectra);
  add_spectra(allSpectra->kxkyt_spectra);
  add_spectra(allSpectra->zt_spectra);
}

void Phi2Diagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  // compute |Phi|**2(ky, kx, z, t)
  Phi2_summand <<<dG, dB>>> (tmpf, f->phi, geo_->vol_fac); 	
  // compute and write spectra of |Phi|**2
  write_spectra(tmpf);

  // get Phi**2(t) data
  float *phi2 = spectraList[0]->get_data();

  if(grids_->iproc==0) {
    printf ("Phi**2 = %.3e   ", phi2[0]);
  }
}

WphiDiagnostic::WphiDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
{
  nc_type = NC_FLOAT;
  pars_ = pars;
  grids_ = grids;
  geo_ = geo;
  ncdf_ = ncdf;
  varname = "Wphi";
  isMoments = false;
  set_kernel_dims();
  nc_group = ncdf_->nc_diagnostics->spectra;

  add_spectra(allSpectra->st_spectra);
  add_spectra(allSpectra->kxst_spectra);
  add_spectra(allSpectra->kyst_spectra);
  add_spectra(allSpectra->kxkyst_spectra);
  add_spectra(allSpectra->zst_spectra);
}

void WphiDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is + grids_->is_lo;
    float rho2s = pars_->species_h[is_glob].rho2;
    Wphi_summand <<<dG, dB>>> (&tmpf[grids_->NxNycNz*is], f->phi, geo_->vol_fac, geo_->kperp2, rho2s); 	
  }
  write_spectra(tmpf);
}

WgDiagnostic::WgDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
{
  nc_type = NC_FLOAT;
  pars_ = pars;
  grids_ = grids;
  geo_ = geo;
  ncdf_ = ncdf;
  varname = "Wg";
  isMoments = true;
  set_kernel_dims();
  nc_group = ncdf_->nc_diagnostics->spectra;

  add_spectra(allSpectra->st_spectra);
  add_spectra(allSpectra->kxst_spectra);
  add_spectra(allSpectra->kyst_spectra);
  add_spectra(allSpectra->kxkyst_spectra);
  add_spectra(allSpectra->zst_spectra);
  add_spectra(allSpectra->lmst_spectra);
}

void WgDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is + grids_->is_lo;
    float nt = pars_->species_h[is_glob].nt;
    Wg_summand <<<dG, dB>>> (&tmpG[grids_->NxNycNz*grids_->Nmoms*is], G[is]->G(), geo_->vol_fac, nt);
  }
  write_spectra(tmpG);
}

HeatFluxDiagnostic::HeatFluxDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
{
  nc_type = NC_FLOAT;
  pars_ = pars;
  grids_ = grids;
  geo_ = geo;
  ncdf_ = ncdf;
  varname = "HeatFlux";
  isMoments = false;
  if(grids_->m_lo>0) skipWrite = true; // procs with higher hermites will have nonsense 
                                       // heat flux data, so skip the write from these procs
  set_kernel_dims();
  nc_group = ncdf_->nc_diagnostics->spectra;

  add_spectra(allSpectra->st_spectra);
  add_spectra(allSpectra->kxst_spectra);
  add_spectra(allSpectra->kyst_spectra);
  add_spectra(allSpectra->kxkyst_spectra);
  add_spectra(allSpectra->zst_spectra);
}

void HeatFluxDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is + grids_->is_lo;
    float rho2s = pars_->species_h[is_glob].rho2;
    float p_s = pars_->species_h[is_glob].nt;
    float vts = pars_->species_h[is_glob].vt;
    heat_flux_summand <<<dG, dB>>> (&tmpf[grids_->NxNycNz*is], f->phi, f->apar, G[is]->G(), grids_->ky,  geo_->flux_fac, geo_->kperp2, rho2s, p_s, vts); 	
  }
  write_spectra(tmpf);

  // get Q(t) data
  float *fluxes = spectraList[0]->get_data();

  if(!skipWrite) {
    for (int is=0; is<grids_->Nspecies; is++) {
      int is_glob = is + grids_->is_lo;
      const char *spec_string = pars_->species_h[is_glob].type == 1 ? "e" : "i";
      printf ("Q_%s = %.3e   ", spec_string, fluxes[is]);
    }
  }
}

ParticleFluxDiagnostic::ParticleFluxDiagnostic(Parameters* pars, Grids* grids, Geometry* geo, NetCDF* ncdf, AllSpectraCalcs* allSpectra)
{
  nc_type = NC_FLOAT;
  pars_ = pars;
  grids_ = grids;
  geo_ = geo;
  ncdf_ = ncdf;
  varname = "ParticleFlux";
  isMoments = false;
  if(grids_->m_lo>0) skipWrite = true; // procs with higher hermites will have nonsense 
                                       // particle flux data, so skip the write from these procs
  set_kernel_dims();
  nc_group = ncdf_->nc_diagnostics->spectra;

  add_spectra(allSpectra->st_spectra);
  add_spectra(allSpectra->kxst_spectra);
  add_spectra(allSpectra->kyst_spectra);
  add_spectra(allSpectra->kxkyst_spectra);
  add_spectra(allSpectra->zst_spectra);
}

void ParticleFluxDiagnostic::calculate_and_write(MomentsG** G, Fields* f, float* tmpG, float* tmpf)
{
  for(int is=0; is<grids_->Nspecies; is++) {
    int is_glob = is + grids_->is_lo;
    float rho2s = pars_->species_h[is_glob].rho2;
    float n_s = pars_->nspec>1 ? pars_->species_h[is].dens : 0.;
    float vts = pars_->species_h[is_glob].vt;
    particle_flux_summand <<<dG, dB>>> (&tmpf[grids_->NxNycNz*is], f->phi, f->apar, G[is]->G(), grids_->ky,  geo_->flux_fac, geo_->kperp2, rho2s, n_s, vts); 	
  }
  write_spectra(tmpf);

  // get Gam(t) data
  float *fluxes = spectraList[0]->get_data();

  if(!skipWrite) {
    for (int is=0; is<grids_->Nspecies; is++) {
      int is_glob = is + grids_->is_lo;
      const char *spec_string = pars_->species_h[is_glob].type == 1 ? "e" : "i";
      printf ("Gam_%s = %.3e   ", spec_string, fluxes[is]);
    }
  }
}

GrowthRateDiagnostic::GrowthRateDiagnostic(Parameters* pars, Grids* grids, NetCDF* ncdf)
{
  nc_type = NC_FLOAT;
  pars_ = pars;
  grids_ = grids;
  ncdf_ = ncdf;
  varname = "omega_kxkyt";
  nc_group = ncdf_->nc_diagnostics->diagnostics_id;
  ndim = 4;

  dims[0] = ncdf_->nc_dims->time;
  dims[1] = ncdf_->nc_dims->ky;
  dims[2] = ncdf_->nc_dims->kx;
  dims[3] = ncdf_->nc_dims->ri;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Naky;
  count[2] = grids->Nakx;
  count[3] = 2;

  N = grids->NxNyc;
  Nwrite = grids->Nakx*grids->Naky*2;

  int retval;
  if (retval = nc_def_var(nc_group, varname.c_str(), nc_type, ndim, dims, &varid)) ERR(retval);
  if (retval = nc_var_par_access(nc_group, varid, NC_COLLECTIVE)) ERR(retval);

  cudaMalloc (&omg_d, sizeof(cuComplex) * N);
  omg_h = (cuComplex*) malloc  (sizeof(cuComplex) * N);
  cpu = (float*) malloc  (sizeof(float) * Nwrite);
}

GrowthRateDiagnostic::~GrowthRateDiagnostic()
{
  cudaFree(omg_d);
  free(omg_h);
  free(cpu);
}

// need separate calculate and write methods for growth rates, 
// so that can calculate every step but write less often
void GrowthRateDiagnostic::calculate(Fields* fields, Fields* fields_old, double dt)
{
  int nt = min(512, grids_->NxNyc) ;
  growthRates <<< 1 + (grids_->NxNyc-1)/nt, nt >>> (fields->phi, fields_old->phi, dt, omg_d);
  fields_old->copyPhiFrom(fields);
}

void GrowthRateDiagnostic::write()
{
  // write to ncdf
  CP_TO_CPU(omg_h, omg_d, sizeof(cuComplex)*N);
  dealias_and_reorder(omg_h, cpu);
  
  int retval;
  start[0] = ncdf_->nc_grids->time_index;
  if (retval=nc_put_vara(nc_group, varid, start, count, cpu)) ERR(retval);

  // print to screen
  int Nx = grids_->Nx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;

  printf("\nky\tkx\t\tomega\t\tgamma\n");

  for(int j=0; j<Naky; j++) {
    for(int i= 1 + 2*Nx/3; i<Nx; i++) {
      int index = j + Nyc*i;
      printf("%.4f\t%.4f\t\t%.6f\t%.6f",  grids_->ky_h[j], grids_->kx_h[i], omg_h[index].x, omg_h[index].y);
      printf("\n");
    }
    for(int i=0; i < 1 + (Nx-1)/3; i++) {
      int index = j + Nyc*i;
      if(index!=0) {
        printf("%.4f\t%.4f\t\t%.6f\t%.6f", grids_->ky_h[j], grids_->kx_h[i], omg_h[index].x, omg_h[index].y);
        printf("\n");
      } else {
        printf("%.4f\t%.4f\n", grids_->ky_h[j], grids_->kx_h[i]);
      }
    }
    if (Nx>1) printf("\n");
  }

}

void GrowthRateDiagnostic::dealias_and_reorder(cuComplex* fold, float* fnew)
{
  int Nx   = grids_->Nx;
  int Nakx = grids_->Nakx;
  int Naky = grids_->Naky;
  int Nyc  = grids_->Nyc;

  int NK = grids_->Nakx/2;
 
  int it = 0;
  int itp = it + NK;
  for (int ik=0; ik<Naky; ik++) {
    int Qp = itp + ik*Nakx;
    int Rp = ik  + it*Nyc;
    fnew[2*Qp  ] = fold[Rp].x;
    fnew[2*Qp+1] = fold[Rp].y;
  }

  for (int it = 1; it < NK+1; it++) {
    int itp = NK + it;
    int itn = NK - it;
    int itm = Nx - it;
    for (int ik=0; ik<Naky; ik++) {
      int Qp = itp + ik*Nakx;
      int Rp = ik  + it*Nyc;

      int Qn = itn + ik*Nakx;
      int Rm = ik  + itm*Nyc;
      fnew[2*Qp  ] = fold[Rp].x;
      fnew[2*Qp+1] = fold[Rp].y;

      fnew[2*Qn  ] = fold[Rm].x;
      fnew[2*Qn+1] = fold[Rm].y;
    }
  }
}
