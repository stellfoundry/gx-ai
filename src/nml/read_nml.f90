subroutine namelistRead(ncid, runname) bind(c, name='namelistRead')

  ! before calling, open a netcdf file, define the input variables
  ! call nc_enddef, and pass the ncid and runname to this routine

  use, intrinsic :: iso_c_binding, only: c_int
  
  implicit none
  character(200), intent(in)       :: runname 
  integer (kind=c_int), intent(in) :: ncid

  private

  logical :: debug, restart, nonlinear_mode, slab, const_curv
  logical :: hyper, hypercollisions, write_omega, write_phi
  logical :: write_hermite_spectrum, zero_order_nonlin_flr_only
  logical :: no_nonlin_cross_terms, non_nonlin_dens_cross_term
  logical :: init_single, snyder_electrons, forcing_init
  integer :: ntheta, nperiod, nx, ny, nhermite, nlaguerre
  integer :: smith_par_q, smith_perp_q, nstep, nwrite, navg, nsave
  integer :: jtwist, nspec, iphi00, igeo, p_hyper, p_hyper_l, p_hyper_m
  integer :: hermite_spectrum_avg_cutoff, ikx_single, iky_single
  integer :: forcing_index
  real :: x0, y0, delt, zp, tite, fphi, fapar, fbpar, beta, cfl
  real :: rhoc, eps, shat, qinp, shift, akappa, akappri, tri, tripri
  real :: Rmaj, beta_prime_input, s_hat_input, drhodpsi, kxfac
  real :: phi_ext, nu_hyper, d_hyper, nu_hyper_l, nu_hyper_m
  real :: init_amp, kpar_init, forcing_amp
  character(20) :: boundary_option, closure_model, scheme, geofile
  character(20) :: source_option, init, forcing_type

  namelist /gx/ debug, ntheta, nperiod, nx, ny, x0, y0, boundary_option, &
       nhermite, nlaguerre, closure_model, smith_par_q, smith_perp_q, &
       nstep, nwrite, delt, navg, nsave, scheme, restart, &
       zp, jtwist, nspec, tite, iphi00, fphi, fapar, fbpar, beta, &
       nonlinear_mode, cfl, &
       igeo, geofile, slab, const_curv, rhoc, eps, shat, qinp, shift, &
       akappa, akappri, tri, tripri, Rmaj, beta_prime_input, s_hat_input, &
       drhodpsi, kxfac, source_option, phi_ext, &
       hyper, nu_hyper, p_hyper, d_hyper, &
       hypercollisions, nu_hyper_l, nu_hyper_m, p_hyper_l, p_hyper_m, &
       init_amp, init, write_omega, write_phi, write_hermite_spectrum, &       
       hermite_spectrum_avg_cutoff, zero_order_nonlin_flr_only, &
       no_nonlin_cross_terms, no_nonlin_dens_cross_term, &
       init_single, iky_single, ikx_single, kpar_init, snyder_electrons, &       
       forcing_init, forcing_type, forcing_amp, forcing_index
 ! netcdf_restart, append_old, nlpm, inlpm, dnlpm, dnlpm_dens, dnlpm_tprp, &
 ! low_b_all, iflr, avail_cpu_time, margin_cpu_time, iso_shear, scan_type, &
 ! scan_number, zero_restart_avg, no_zderiv_covering, no_zderiv, zderiv_loop, &
 ! icovering, no_landau_damping, turn_off_gradients_test, write_netcdf, &
 ! restartfile, no_zonal_nlpm, dorland_qneut, stationary_ions, new_nlpm, &
 ! hammett_nlpm_interference, higher_order_moments, nlpm_zonal_only, &
 ! nlpm_vol_avg, nlpm_abs_sgn, nlpm_kxdep, nlpm_nlps, nlpm_cutoff_avg, &
 ! nlpm_zonal_kx1_only, smagorinsky, dorland_nlpm, dorland_nlpm_phase, &
 ! dorland_phase_complex, dorland_phase_ifac, nlpm_option, dnlpm_max, tau_nlpm


  open(unit=100, file='runname', status='old')
  read(unit=100, nml = gx)
  close(unit=100)

  ! To do the writes to the netcdf file, need to find the handles
  ! for each variable. 

  ! write the input choices to the netcdf file, sync the file 
  ! and return

endsubroutine namelistRead
