#pragma once
#include "cufft.h"
#include "cufftXt.h"
#include "species.h"

__device__ unsigned int get_id1(void);
__device__ unsigned int get_id2(void);
__device__ unsigned int get_id3(void);

__host__ __device__ float factorial(int m);
__device__ float Jflr(int l, float b, bool enforce_JL_0=true);
__device__ float Jfac(int l, float b);

__host__ __device__ float g0(float b);
__host__ __device__ float g1(float b);
__host__ __device__ float sgam0 (float b);

__host__ __device__ bool operator>(cuComplex f, cuComplex g);
__host__ __device__ bool operator<(cuComplex f, cuComplex g);

__host__ __device__ cuComplex operator+(cuComplex f, cuComplex g); 
__host__ __device__ cuComplex operator+(cuComplex g, float f);
__host__ __device__ cuComplex operator+(float f, cuComplex g);
__host__ __device__ cuComplex operator-(cuComplex f, cuComplex g);
__host__ __device__ cuComplex operator-(cuComplex f);

__host__ __device__ cuComplex operator*(float scaler, cuComplex f) ;
__host__ __device__ cuComplex operator*(cuComplex f, float scaler); 
__host__ __device__ cuComplex operator*(cuComplex f, cuComplex g);
__host__ __device__ cuDoubleComplex operator*(cuDoubleComplex f, cuDoubleComplex g);

__host__ __device__ cuComplex operator/(cuComplex f, float scaler);
__host__ __device__ cuComplex operator/(cuComplex f, cuComplex g) ;
__host__ __device__ cuDoubleComplex operator/(cuDoubleComplex f, cuDoubleComplex g) ;

__global__ void acc_scaled_kernel(cuComplex* res,
				  double c1, const cuComplex* m1);

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2);

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2,
				  double c3, const cuComplex* m3);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2, 
				  double c3, const cuComplex* m3,
				  double c4, const cuComplex* m4);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2, 
				  double c3, const cuComplex* m3,
				  double c4, const cuComplex* m4,
				  double c5, const cuComplex* m5);

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2, bool bdum);

__global__ void add_scaled_kernel(cuComplex* res,
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2,
				  double c3, const cuComplex* m3, bool bdum);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2, 
				  double c3, const cuComplex* m3,
				  double c4, const cuComplex* m4, bool bdum);

__global__ void add_scaled_kernel(cuComplex* res, 
				  double c1, const cuComplex* m1,
				  double c2, const cuComplex* m2, 
				  double c3, const cuComplex* m3,
				  double c4, const cuComplex* m4,
				  double c5, const cuComplex* m5, bool bdum);

__global__ void scale_kernel(cuComplex* res, double s);
__global__ void scale_kernel(cuComplex* res, cuComplex s);
__global__ void scale_singlemom_kernel(cuComplex* res, cuComplex* m, cuComplex s);

// Should we have some eqfix options in the singlemom_kernels? 

__global__ void add_section(cuComplex *res, const cuComplex *tmp, int ntot);

__global__ void add_scaled_singlemom_kernel(cuComplex* res,
					    double c1, const cuComplex* m1,
					    double c2, const cuComplex* m2);
__global__ void add_scaled_singlemom_kernel(cuComplex* res, 
					    double c1, const cuComplex* m1,
					    double c2, const cuComplex* m2,
					    double c3, const cuComplex* m3);
__global__ void add_scaled_singlemom_kernel(cuComplex* res,
					    cuComplex c1, cuComplex* m1,
					    cuComplex c2, cuComplex* m2);

__global__ void reality_kernel(cuComplex* g, int N);

__device__ int get_ikx(int idx);
__device__ bool unmasked(int idx, int idy);
__device__ bool   masked(int idx, int idy);

__global__ void Tbar(cuComplex* t_bar, const cuComplex* g, const cuComplex* phi, const float *kperp2);
		     
__global__ void growthRates(const cuComplex *phi, const cuComplex *phiOld, double dt, cuComplex *omega);

__global__ void J0phiToGrid(cuComplex* J0phi, const cuComplex* phi, const float* b,
			    const float* muB, float rho2_s);

__global__ void acc(float *a, const float *b);

__global__ void bracket(float* g_res,
			const float* dg_dx, const float* dJ0phi_dy,
			const float* dg_dy, const float* dJ0Phi_dx, float kxfac);

__global__ void castDoubleToFloat (const cuDoubleComplex *array_d, cuComplex *array_f, int size);

__global__ void beer_toroidal_closures(const cuComplex* g, cuComplex* gRhs,
				       const float* omegad, const cuComplex* nu, const specie *species);

__global__ void smith_perp_toroidal_closures(const cuComplex* g, cuComplex* gRhs,
					     const float* omegad, const cuComplex* Aclos, int q, specie *species);

__global__ void stirring_kernel(const cuComplex force, cuComplex *moments, int forcing_index);

__global__ void fieldlineaverage(cuComplex *favg, cuComplex *df, const cuComplex *f, const float *volJac);

__global__ void W_summand(float *G2, const cuComplex* g, const float* volJac, const specie *species);

__global__ void vol_summand(float *rmom, const cuComplex* f, const cuComplex* g, const float* volJac);

__global__ void get_pzt (float* primary, float* secondary, float* tertiary, const cuComplex* phi, const cuComplex* tbar);
  
__global__ void Wphi_scale(float* p2, float alpha);

__global__ void Wphi2_summand(float *p2, const cuComplex *phi, const float *volJac);
  
__global__ void Wphi_summand(float* p2, const cuComplex* phi, const float* volJac, const float* kperp2, float rho2_s);
  
__global__ void heat_flux_summand(float* qflux, const cuComplex* phi, const cuComplex* g, const float* ky, 
				  const float* flxJac, const float *kperp2, float rho2_s);

__global__ void init_kperp2(float* kperp2, const float* kx, const float* ky,
			    const float* gds2, const float* gds21, const float* gds22,
			    const float* bmagInv, float shat);

__global__ void init_omegad(float* omegad, float* cv_d, float* gb_d, const float* kx, const float* ky,
			    const float* cv, const float* gb, const float* cv0, const float* gb0, float shat);

__global__ void calc_bgrad(float* bgrad, const float* bgrad_temp, const float* bmag, float scale);

__device__ cuComplex i_kx(void *dataIn, size_t offset, void *kxData, void *sharedPtr);
__device__ cuComplex i_ky(void *dataIn, size_t offset, void *kyData, void *sharedPtr);
__device__ void mask_and_scale(void *dataOut, size_t offset, cufftComplex element, void *data, void * sharedPtr);

extern __managed__ cufftCallbackLoadC i_kx_callbackPtr;
extern __managed__ cufftCallbackLoadC i_ky_callbackPtr;
extern __managed__ cufftCallbackStoreC mask_and_scale_callbackPtr;
  
__device__ void i_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void abs_kz(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void i_kz_1d(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);

extern __managed__ cufftCallbackStoreC i_kz_callbackPtr;
extern __managed__ cufftCallbackStoreC i_kz_1d_callbackPtr;
extern __managed__ cufftCallbackStoreC abs_kz_callbackPtr;

__global__ void kInit(float* kx, float* ky, float* kz, const float X0, const float Y0, const int Zp);

__global__ void qneut(cuComplex* Phi, const cuComplex* g, const float* kperp2, const specie* species);
__global__ void ampere(cuComplex* Apar, const cuComplex* gu, const float* kperp2, const specie* species, float beta);

__global__ void real_space_density(cuComplex* nbar, const cuComplex* g, const float *kperp2, const specie *species);

//__global__ void qneut_fieldlineaveraged(cuComplex *Phi, const cuComplex *nbar, const float *PhiAvgDenom, 
//					const float *kperp2, const float *jacobian,
//					const specie *species, const float ti_ov_te, float *work);

__global__ void qneutAdiab_part1(cuComplex* PhiAvgNum_tmp, const cuComplex* nbar,
				 const float* kperp2, const float* jacobian, const specie* species, const float ti_ov_te);

__global__ void qneutAdiab_part2(cuComplex* Phi, const cuComplex* PhiAvgNum_tmp, const cuComplex* nbar,
				 const float* PhiAvgDenom, const float* kperp2, 
				 const specie* species, const float ti_ov_te);

__global__ void calc_phiavgdenom(float* PhiAvgDenom, const float* kperp2,
				 const float* jacobian, const specie* species, const float ti_ov_te);

__global__ void add_source(cuComplex* f, const float source);

__global__ void qneutAdiab(cuComplex* Phi, const cuComplex* nbar,
			   const float* kperp2, const specie* species, float ti_ov_te);

__global__ void linkedCopy(const cuComplex* G, cuComplex* G_linked, int nLinks, int nChains,
			   const int* ikx, const int* iky, int nMoms);

__global__ void linkedCopyBack(const cuComplex* G_linked, cuComplex* G, int nLinks, int nChains,
			       const int* ikx, const int* iky, int nMoms);

__device__ void i_kzLinked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__device__ void abs_kzLinked(void *dataOut, size_t offset, cufftComplex element, void *kzData, void *sharedPtr);
__global__ void init_kzLinked(float* kz, int nLinks);

extern __managed__ cufftCallbackStoreC i_kzLinked_callbackPtr;
extern __managed__ cufftCallbackStoreC abs_kzLinked_callbackPtr;

__global__ void nlks(float *res, const float *Gy, const float *dG);
__global__ void rhs_ks (const cuComplex *G, cuComplex *GRhs, float *ky);
__global__ void rhs_linear(const cuComplex *g, const cuComplex* phi,
			   const cuComplex* upar_bar, const cuComplex* uperp_bar, const cuComplex* t_bar,
			   const float* b, const float* cv_d, const float* gb_d, const float* bgrad,
			   const float* ky, const specie* s, cuComplex* rhs_par, cuComplex* rhs);

__global__ void conservation_terms(cuComplex* upar_bar, cuComplex* uperp_bar,
				   cuComplex* t_bar, const cuComplex* G, const cuComplex* phi,
				   const float *b, const specie* species);

__global__ void hypercollisions(const cuComplex* g, const float nu_hyper_l, const float nu_hyper_m,
				const int p_hyper_l, const int p_hyper_m, cuComplex* rhs);



