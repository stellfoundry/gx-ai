#include "spectra_diagnostics.h"

Spectra::~Spectra()
{
  cudaFree(data);
  free(tmp);
  free(cpu);
  delete field_reduce;
  delete moments_reduce;
}

void Spectra::allocate()
{
  cudaMalloc (&data, sizeof(float) * N);
  tmp = (float*) malloc  (sizeof(float) * N);
  cpu = (float*) malloc  (sizeof(float) * Nwrite);
}

int Spectra::define_variable(string varstem, int nc_group)
{
  int varid, retval;
  if (retval = nc_def_var(nc_group, (varstem + tag).c_str(), NC_FLOAT, ndim, dims, &varid)) ERR(retval);
  if (retval = nc_var_par_access(nc_group, varid, NC_COLLECTIVE)) ERR(retval);
  return varid;
}

void Spectra::write(float *fullData, int varid, int nc_group, bool isMoments)
{
  if(isMoments) moments_reduce->Sum(fullData, data); 
  else field_reduce->Sum(fullData, data); 
  CP_TO_CPU(tmp, data, sizeof(float)*N);
  if(N!=Nwrite) dealias_and_reorder(tmp, cpu);
  else cpu = tmp;
  
  int retval;
  if (retval=nc_put_vara(nc_group, varid, start, count, cpu)) ERR(retval);
  start[0] += 1;
}

Spectra_kyst::Spectra_kyst(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_kyst";
  ndim = 3;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->species;
  dims[2] = nc_dims->ky;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nspecies;
  count[2] = grids->Naky;

  start[1] = grids->is_lo;

  field_reduce = new Grid_Species_Reduce(grids, KY);
  moments_reduce = new All_Reduce(grids, KY);

  N = grids->Nyc*grids->Nspecies;
  Nwrite = grids->Naky*grids->Nspecies; // only write de-aliased modes

  allocate();
}

void Spectra_kyst::dealias_and_reorder(float *fold, float *fnew)
{
  for (int is = 0; is < grids_->Nspecies; is++) {
    for (int ik = 0; ik < grids_->Naky; ik++) {
      fnew[ik + is*grids_->Naky] = fold[ik + is*grids_->Nyc];
    }
  }
}

Spectra_kxkyst::Spectra_kxkyst(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_kxkyst";
  ndim = 4;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->species;
  dims[2] = nc_dims->ky;
  dims[3] = nc_dims->kx;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Nspecies;
  count[2] = grids->Naky;
  count[3] = grids->Nakx;

  start[1] = grids->is_lo;

  field_reduce = new Grid_Species_Reduce(grids, KXKY);
  moments_reduce = new All_Reduce(grids, KXKY);

  N = grids->Nx*grids->Nyc*grids->Nspecies;
  Nwrite = grids->Nakx*grids->Naky*grids->Nspecies; // only write de-aliased modes

  allocate();
}

void Spectra_kxkyst::dealias_and_reorder(float *fold, float *fnew)
{
  int NK = grids_->Nakx/2;
  int NX = grids_->Nx; 
  for (int is = 0; is < grids_->Nspecies; is++) {
    int it = 0;
    int itp = it + NK;
    for (int ik = 0; ik < grids_->Naky; ik++) {
      int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
      int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc *grids_->Nx;
      fnew[Qp] = fold[Rp];
    }	
    for (int it = 1; it < NK+1; it++) {
      int itp = NK + it;
      int itn = NK - it;
      int itm = NX - it;
      
      for (int ik = 0; ik < grids_->Naky; ik++) {

        int Qp = itp + ik*grids_->Nakx + is*grids_->Naky*grids_->Nakx;
        int Rp = ik  + it*grids_->Nyc  + is*grids_->Nyc * NX;

        int Qn = itn + ik *grids_->Nakx + is*grids_->Naky*grids_->Nakx;
        int Rm = ik  + itm*grids_->Nyc  + is*grids_->Nyc * NX;

        fnew[Qp] = fold[Rp];
        fnew[Qn] = fold[Rm];
      }
    }
  }
}

Spectra_kyt::Spectra_kyt(Grids* grids, NcDims *nc_dims)
{
  grids_ = grids;
  tag = "_kyt";
  ndim = 2;

  dims[0] = nc_dims->time;
  dims[1] = nc_dims->ky;

  count[0] = 1; // each write is a single time slice
  count[1] = grids->Naky;

  field_reduce = new Grid_Reduce(grids, KY);
  moments_reduce = new All_Reduce(grids, KY);

  N = grids->Nyc;
  Nwrite = grids->Naky; // only write de-aliased modes

  allocate();
}

void Spectra_kyt::dealias_and_reorder(float *fold, float *fnew)
{
  for (int ik = 0; ik < grids_->Naky; ik++) {
    fnew[ik] = fold[ik];
  }
}

SpectraDiagnostic::SpectraDiagnostic(string varname, int nc_group, int nc_type)
  : varname(varname), nc_group(nc_group), nc_type(nc_type)  
{
}

void SpectraDiagnostic::add_spectra(Spectra *spectra)
{
  spectraList.push_back(*spectra);
  int varid = spectra->define_variable(varname, nc_group);
  spectraIds.push_back(varid);
}

void SpectraDiagnostic::write(float* data, bool isMoments)
{
  for(int i=0; i<spectraList.size(); i++) {
    spectraList[i].write(data, spectraIds[i], nc_group, isMoments);
  }
}
