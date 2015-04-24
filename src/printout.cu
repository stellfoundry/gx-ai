#include "standard_headers.h"
#include "printout.h"

void print_initial_parameter_summary(everything_struct * ev){
  printf("\nNx=%d  Ny=%d  Nz=%d  X0=%g  Y0=%g  Zp=%d   igeo=%d\n", Nx, Ny, Nz, X0, Y0, Zp, igeo);
  printf("tprim=%g  fprim=%g\njtwist=%d   nSpecies=%d   cfl=%f\n", species[ION].tprim, species[ION].fprim,jtwist,nSpecies,cfl);
  printf("temp=%g  dens=%g nu_ss=%g  inlpm=%d  dnlpm=%f\n", species[ION].temp, species[ION].dens,species[ION].nu_ss, inlpm, dnlpm);
  printf("shat=%g  eps=%g  qsf=%g  rmaj=%g  g_exb=%g\n", shat, eps, qsf, rmaj, g_exb);
  printf("rgeo=%g  akappa=%g  akappapri=%g  tri=%g  tripri=%g\n", r_geo, akappa, akappri, tri, tripri);
  printf("asym=%g  asympri=%g  beta_prime_input=%g  rhoc=%g\n", asym, asympri, beta_prime_input, rhoc);
  if(NLPM && nlpm_kxdep) printf("USING NEW KX DEPENDENCE IN COMPLEX DORLAND NLPM EXPRESSION\n");
  if(nlpm_nlps) printf("USING NEW NLPS-style NLPM\n");

}

void print_cuda_properties(everything_struct * ev){
    int ct, dev;
    int driverVersion =0, runtimeVersion=0;
    struct cudaDeviceProp prop;

    cudaGetDeviceCount(&ct);
    printf("Device Count: %d\n",ct);

    cudaGetDevice(&dev);
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("Driver Version / Runtime Version: %d.%d / %d.%d\n", driverVersion/1000, driverVersion%100,runtimeVersion/1000,runtimeVersion%100);
    cudaGetDeviceProperties(&prop,dev);
    printf("Device Name: %s\n", prop.name);
    printf("Global Memory (bytes): %lu\n", (unsigned long)prop.totalGlobalMem);
    printf("Shared Memory per Block (bytes): %lu\n", (unsigned long)prop.sharedMemPerBlock);
    printf("Registers per Block: %d\n", prop.regsPerBlock);
    printf("Warp Size (threads): %d\n", prop.warpSize); 
    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("Max Size of Block Dimension (threads): %d * %d * %d\n", prop.maxThreadsDim[0], 
	   prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Size of Grid Dimension (blocks): %d * %d * %d\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
}

void print_final_summary(everything_struct * ev, FILE * outfile){
    printf("\nNx=%d  Ny=%d  Nz=%d  X0=%g  Y0=%g  Zp=%d\n", Nx, Ny, Nz, X0, Y0,Zp);
    printf("tprim=%g  fprim=%g\njtwist=%d   nSpecies=%d   cfl=%f\n", species[ION].tprim, species[ION].fprim,jtwist,nSpecies,cfl);
    printf("shat=%g  eps=%g  qsf=%g  rmaj=%g  g_exb=%g\n", shat, eps, qsf, rmaj, g_exb);
    if(LINEAR) printf("[Linear]\t");
    else printf("[Nonlinear]\t");
    if(NO_ZDERIV) printf("[No zderiv]\t");
    if(NO_ZDERIV_COVERING) printf("[No zderiv_covering]\t");
    if(SLAB) printf("[Slab limit]\t");
    if(varenna) printf("[varenna: ivarenna=%d]\t", ivarenna);
    if(CONST_CURV) printf("[constant curvature]\t");
    if(RESTART) printf("[restart]\t");
    if(NLPM && nlpm_zonal_kx1_only) {
       printf("[Nonlinear Phase Mixing: inlpm=%d, dnlpm=%f, Phi2_zf(kx=1) only]\t", inlpm, dnlpm);
    }
    if(SMAGORINSKY) printf("[Smagorinsky Diffusion]\t");
    if(HYPER && isotropic_shear) printf("[HyperViscocity: D_hyper=%f, isotropic_shear]\t", D_hyper);
    if(HYPER && !isotropic_shear) printf("[HyperViscocity: D_hyper=%f, anisotropic_shear]\t", D_hyper);
    if(no_landau_damping) printf("[No landau damping]\t");
    if(turn_off_gradients_test) printf("[Gradients turned off halfway through the run]\t");
    
    printf("\n\n");
    
    
    fprintf(outfile,"\nNx=%d  Ny=%d  Nz=%d  X0=%g  Y0=%g  Zp=%d\n", Nx, Ny, Nz, X0, Y0,Zp);
    fprintf(outfile,"tprim=%g  fprim=%g\njtwist=%d   nSpecies=%d   cfl=%f\n", species[ION].tprim, species[ION].fprim,jtwist,nSpecies,cfl);
    fprintf(outfile,"shat=%g  eps=%g  qsf=%g  rmaj=%g  g_exb=%g\n", shat, eps, qsf, rmaj, g_exb);
    if(LINEAR) fprintf(outfile,"[Linear]\t");
    else fprintf(outfile,"[Nonlinear]\t");
    if(NO_ZDERIV) fprintf(outfile,"[No zderiv]\t");
    if(NO_ZDERIV_COVERING) fprintf(outfile,"[No zderiv_covering]\t");
    if(SLAB) fprintf(outfile,"[Slab limit]\t");
    if(varenna) fprintf(outfile,"[varenna: ivarenna=%d]\t", ivarenna);
    if(CONST_CURV) fprintf(outfile,"[constant curvature]\t");
    if(RESTART) fprintf(outfile,"[restart]\t");
    if(NLPM) fprintf(outfile,"[Nonlinear Phase Mixing: inlpm=%d, dnlpm=%f]\t", inlpm, dnlpm);
    if(SMAGORINSKY) fprintf(outfile,"[Smagorinsky Diffusion]\t");
    if(HYPER && isotropic_shear) fprintf(outfile, "[HyperViscocity: D_hyper=%f, isotropic_shear]\t", D_hyper);
    if(HYPER && !isotropic_shear) fprintf(outfile, "[HyperViscocity: D_hyper=%f, anisotropic_shear]\t", D_hyper);
    
    fprintf(outfile, "\n\n");
}
