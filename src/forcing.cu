#include "forcing.h"
#include "get_error.h"
#include "device_funcs.h"
#include "cuda_constants.h"

__global__ void stirring_kernel(cuComplex force, cuComplex *moments, int forcing_index);
void generate_random_numbers(float *random_real, float *random_imag, float forcing_amp_, float dt);

/* The following knobs in forcing_knobs need to be set to turn on forcing:
forcing_init = T
forcing_amp = 1
forcing_type = "Kz" or "KzImpulse"
*/

// Set forcing_type to "Kz"
KzForcing::KzForcing(Parameters *pars) : pars_(pars)
{
  forcing_amp_ = pars_->forcing_amp;
  printf("forcing_amp = %f \n", forcing_amp_);
}

KzForcing::~KzForcing()
{
}

// Langevin forcing done at the end of each timestep in timestepper's advance method
void KzForcing::stir(MomentsG *G) {
  float random_real, random_imag;
  generate_random_numbers (&random_real, &random_imag, forcing_amp_, pars_->dt);

  random_force.x = random_real;
  random_force.y = random_imag;

  stirring_kernel <<<1,1>>> (random_force, G->G(0,0,0), pars_->forcing_index);
}

genForcing::genForcing(Parameters *pars) : pars_(pars)
{
  forcing_amp_ = pars_->forcing_amp;
}

genForcing::~genForcing()
{
}

void genForcing::stir(MomentsG *G) {
  float random_real, random_imag;
  generate_random_numbers (&random_real, &random_imag, forcing_amp_, pars_->dt);

  random_force.x = random_real;
  random_force.y = random_imag;

  stirring_kernel <<<1,1>>> (random_force, G->G(0,0,0), pars_->forcing_index);
}

// Set forcing_type to "KzImpulse"
KzForcingImpulse::KzForcingImpulse(Parameters *pars) : KzForcing(pars)
{
  stirring_done = false;
}

// Langevin forcing done once at the first timestep
void KzForcingImpulse::stir(MomentsG *G) {
  if(stirring_done) {
    return;
  }

  float random_real, random_imag;
  generate_random_numbers(&random_real, &random_imag, forcing_amp_, pars_->dt);

  random_force.x = random_real;
  random_force.y = random_imag;

  stirring_kernel <<<1,1>>> (random_force, G->G(0,0,0), pars_->forcing_index);
  stirring_done = true;
}

__global__ void stirring_kernel(cuComplex force, cuComplex *moments, int forcing_index) {
    moments[forcing_index] = moments[forcing_index] + force;
}

void generate_random_numbers(float *random_real, float *random_imag, float forcing_amp_, float dt) {

  // Box-Muller transform to generate random normal variables
  float ran_amp = ( (float) rand()) / ((float) RAND_MAX + 1.0 );

  // dt term in timestepper scheme accounted for in amp
  float amp = 1.0*(sqrt(abs(1.0*(forcing_amp_*dt)*log(ran_amp))));
  float phase = M_PI*(2.0*( (float) rand()) / ((float) RAND_MAX + 1.0 ) -1.0);

  *random_real = amp*cos(phase);
  *random_imag = amp*sin(phase);
}

