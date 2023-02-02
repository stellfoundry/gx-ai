#include "spectra_diagnostics.h"

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
