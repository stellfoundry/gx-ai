#include "forcing.h"
#include "get_error.h"
#include "device_funcs.h"
#include "cuda_constants.h"

__global__ void fill_stirring_kernel(cuComplex *random_force, float *z, float kpar, float Zp, float random_real);

ZForcing::ZForcing(Parameters *pars, Grids *grids, Geometry *geo) :
  pars_(pars), grids_(grids), geo_(geo) 
{
  forcing_amp_ = pars_->forcing_amp;

  dimBlock = 512;
  dimGrid = grids_->NxNycNz/dimBlock.x+1;

  cudaMalloc((void**) &random_force, sizeof(cuComplex)*grids_->NxNycNz);
}

ZForcing::~ZForcing()
{
  cudaFree(random_force);
}

// Langevin forcing done at the end of each timestep in timestepper's advance method
void ZForcing::stir(MomentsG *G) {
  // Box-Muller transform to generate random normal variables
  
  float ran_amp = ( (float) rand()) / ((float) RAND_MAX + 1.0 );
  
  // dt term in timestepper scheme accounted for in amp
  float amp = 1.0*(sqrt(abs(1.0*(forcing_amp_*pars_->dt)*log(ran_amp))));
  float phase = M_PI*(2.0*( (float) rand()) / ((float) RAND_MAX + 1.0 ) -1.0);

  float random_real = amp*cos(phase);
  //float random_imag = amp*sin(phase);

  // fill array with one kicked fourier mode
  fill_stirring_kernel<<<dimGrid, dimBlock>>>(random_force, geo_->z, pars_->kpar_init, pars_->Zp, random_real);

  // apply kick to density
  add_scaled_singlemom_kernel<<<dimGrid, dimBlock>>>(G->G(0,0), 1., G->G(0,0), 1., random_force);
}

// random_force is an array of size NxNycNz and is filled with a cos(kz) dependence perturbation (for one moment)
__global__ void fill_stirring_kernel(cuComplex *random_force, float *z, float kpar, float Zp, float random_real)
{
  unsigned int idxyz = get_id1();
  unsigned int idz = get_id3();

  if (idxyz < nx*nyc*nz && idz < nz) {
    random_force[idxyz].x = random_real*cos(kpar*z[idz]/Zp);
  }
}

