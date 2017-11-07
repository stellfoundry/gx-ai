#include "forcing.h"
#include "get_error.h"
#include "device_funcs.h"
#include "cuda_constants.h"

__global__ void stirring_kernel(cuComplex force, cuComplex *moments);

/* The following knobs in forcing_knobs need to be set to turn on forcing:
forcing_init = "on"
forcing_amp = 1
forcing_type = "Kz" 
*/
KzForcing::KzForcing(Parameters *pars, Grids *grids, Geometry *geo) :
  pars_(pars), grids_(grids), geo_(geo) 
{
  forcing_amp_ = pars_->forcing_amp;
}

KzForcing::~KzForcing()
{
}

// Langevin forcing done at the end of each timestep in timestepper's advance method
void KzForcing::stir(MomentsG *G) {
  // Box-Muller transform to generate random normal variables
  float ran_amp = ( (float) rand()) / ((float) RAND_MAX + 1.0 );

  // dt term in timestepper scheme accounted for in amp
  float amp = 1.0*(sqrt(abs(1.0*(forcing_amp_*pars_->dt)*log(ran_amp))));
  float phase = M_PI*(2.0*( (float) rand()) / ((float) RAND_MAX + 1.0 ) -1.0);

  float random_real = amp*cos(phase);
  float random_imag = amp*sin(phase);

  random_force.x = random_real;
  random_force.y = random_imag;

  stirring_kernel<<<1,1>>>(random_force, G->G(0,0,0));
}

// Stirs density for 1 kz in the local limit
__global__ void stirring_kernel(cuComplex force, cuComplex *moments) {
    /* Use ntheta = 1, nx = 1, ny variable; needs to be generalized later  */
    moments[1] = moments[1] + force;
}
