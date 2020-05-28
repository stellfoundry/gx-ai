subroutine read_nml(c_runname) bind(c, name='read_nml')

  use netcdf
  use iso_c_binding, only: c_char, c_null_char
  
  integer, parameter :: max_spec = 1
  integer, parameter :: sixteen = 16
  integer :: retval, ncid, id_code, id_ri, j, char16_dim
  integer :: id_M, id_L
  integer :: ri = 2

  type :: plasma
     real :: z, mass, vnewk, dens, temp, tprim, fprim, uprim
     character (len=8) :: type
  end type plasma
  
  type (plasma) :: next_sp
  type (plasma), dimension(max_spec) :: sp

  logical :: debug = .false.
  logical :: restart = .false.
  logical :: save_for_restart = .true.
  logical :: secondary = .false.
  logical :: nonlinear_mode = .false.
  logical :: slab = .false.
  logical :: const_curv = .false.
  logical :: hyper = .false.
  logical :: hypercollisions = .false.
  logical :: write_omega = .false.
  logical :: write_fluxes = .false.
  logical :: write_moms = .false.
  logical :: write_phi = .false.
  logical :: write_phi_kpar = .false.
  logical :: write_h_spectrum = .false.
  logical :: write_l_spectrum = .false.
  logical :: write_lh_spectrum = .false.
  logical :: write_pzt = .false.
  logical :: write_rh = .false.
  logical :: init_single = .false.
!  logical :: snyder_electrons = .false.
  logical :: forcing_init = .false.
  logical :: write_spec_v_time = .false.
  logical :: eqfix = .false.
  
  integer :: ntheta = 32
  integer :: nperiod = 1
  integer :: nx = 1
  integer :: ny = 32
  integer :: nhermite = 4
  integer :: nh = -1
  integer :: nlaguerre = 2
  integer :: smith_par_q = 3
  integer :: smith_perp_q = 3
  integer :: nstep = 10000
  integer :: nwrite = 1000
  integer :: navg = 10
  integer :: nsave = 2000000
  integer :: jtwist = -1
  integer :: nspecies = 1
  integer :: iphi00 = 2
  integer :: igeo = 0
  integer :: p_hyper = 2
  integer :: p_hyper_l = 6
  integer :: p_hyper_m = 1
!  integer :: hermite_spectrum_avg_cutoff = 20
  integer :: ikx_single = 0
  integer :: iky_single = 1
  integer :: ikx_fixed = -1
  integer :: iky_fixed = -1
  integer :: forcing_index = 1
  integer :: zp = 1
  integer :: i_share = 8
  
  real :: x0 = 10.0
  real :: y0 = 10.0
  real :: dt = 0.05
  real :: tite = 1.0
  real :: fphi = 1.0
  real :: fapar = 0.0
  real :: fbpar = 0.0
  real :: beta = 0.0
  real :: cfl = 0.1
  real :: rhoc = 0.167
  real :: eps = 0.167
  real :: shat = 0.8
  real :: qinp = 1.4
  real :: shift = 0.0
  real :: akappa = 1.0
  real :: akappri = 0.0
  real :: tri = 0.
  real :: tripri = 0.
  real :: Rmaj = 1.0
  real :: beta_prime_input = 0.0
  real :: s_hat_input = 0.8
  real :: drhodpsi = 1.0
  real :: kxfac = 1.0
  real :: phi_ext = 0.0
  real :: nu_hyper = 1.0
  real :: d_hyper = 0.1
  real :: nu_hyper_l = 1.0
  real :: nu_hyper_m = 1.0
  real :: init_amp = 1.e-5
  real :: kpar_init = 0.
  real :: forcing_amp = 1.0
  real :: scale = 1.0
  
  integer :: id_debug, id_restart, id_nonlinear_mode, id_slab, id_const_curv
  integer :: id_secondary, id_save_for_restart, id_eqfix, id_t_dim
  integer :: id_hyper, id_hypercollisions, id_write_omega, id_write_fluxes
  integer :: id_write_moms, id_write_phi, id_write_phi_kpar, id_write_rh
  integer :: id_write_h_spectrum, id_write_l_spectrum, id_write_lh_spectrum
  integer :: id_init_single, id_forcing_init, id_write_spec_v_time, id_write_pzt
!  integer :: id_snyder_electrons
  integer :: id_ntheta, id_nperiod, id_nx, id_ny, id_nhermite, id_nlaguerre
  integer :: id_smith_par_q, id_smith_perp_q, id_nm, id_nl, id_nspec
  integer :: id_nstep, id_nwrite, id_navg, id_nsave
  integer :: id_jtwist, id_nspecies, id_iphi00, id_igeo
  integer :: id_p_hyper, id_p_hyper_l, id_p_hyper_m
!  integer :: id_hspec_cutoff
  integer :: id_forcing_index, id_ikx_single, id_iky_single
  integer :: id_ikx_fixed, id_iky_fixed
  integer :: id_x0, id_y0, id_dt, id_zp, id_tite, id_i_share
  integer :: id_fphi, id_fapar, id_fbpar, id_beta, id_cfl
  integer :: id_rhoc, id_eps, id_shat, id_qinp, id_shift
  integer :: id_akappa, id_akappri, id_tri, id_tripri
  integer :: id_Rmaj, id_beta_prime_input, id_s_hat_input, id_drhodpsi, id_kxfac
  integer :: id_phi_ext, id_nu_hyper, id_d_hyper, id_nu_hyper_l, id_nu_hyper_m
  integer :: id_init_amp, id_kpar_init, id_forcing_amp, id_scale
  integer :: id_restart_to_file_dum, id_restart_from_file_dum
  integer :: id_boundary_dum, id_closure_model_dum, id_scheme_dum, id_geofile_dum
  integer :: id_source_dum, id_init_field_dum, id_forcing_type_dum
  integer :: id_spec_type_dum, id_spec_z, id_spec_mass, id_spec_vnewk
  integer :: id_spec_dens, id_spec_temp, id_stir_field_dum
  integer :: id_spec_tprim, id_spec_fprim, id_spec_uprim

  integer :: spec_type_dum = -1
  integer :: boundary_dum = -1
  integer :: closure_model_dum = -1
  integer :: scheme_dum = -1
  integer :: geofile_dum = -1
  integer :: restart_to_file_dum = -1
  integer :: restart_from_file_dum = -1
  integer :: source_dum = -1
  integer :: init_field_dum = -1
  integer :: stir_field_dum = -1
  integer :: forcing_type_dum = -1
  
  character(32), dimension(max_spec) :: spec_type = "ion"
  character(32) :: boundary = "linked"
  character(32) :: closure_model = "beer4+2"
  character(32) :: scheme = "k10"
  character(512):: geofile = "eik.out"
  character(512):: restart_to_file = "newsave.nc"
  character(512):: restart_from_file = "oldsave.nc"
  character(32) :: source = "default"
  character(32) :: init_field = "density"
  character(32) :: stir_field = "density"
  character(32) :: forcing_type = "Kz"

  character (kind=c_char, len=1), dimension(256), intent(in) :: c_runname
  character (512) :: runname, nml_file, nc_file, file_header
  character (20)  :: datestamp, timestamp, timezone
  character (2)   :: ci
  character (16)  :: note
  
  namelist /gx/ debug, ntheta, nperiod, nx, ny, x0, y0, boundary, &
       nhermite, nlaguerre, closure_model, smith_par_q, smith_perp_q, &
       nstep, nwrite, dt, navg, nsave, scheme, eqfix, &
       restart, save_for_restart, restart_to_file, restart_from_file, secondary, &
       zp, jtwist, nspecies, tite, iphi00, fphi, fapar, fbpar, beta, &
       nonlinear_mode, cfl, i_share, &
       igeo, geofile, slab, const_curv, rhoc, eps, shat, qinp, shift, &
       akappa, akappri, tri, tripri, Rmaj, beta_prime_input, s_hat_input, &
       drhodpsi, kxfac, source, phi_ext, &
       hyper, nu_hyper, p_hyper, d_hyper, stir_field, &
       hypercollisions, nu_hyper_l, nu_hyper_m, p_hyper_l, p_hyper_m, &
       init_amp, init_field, write_omega, write_fluxes, write_moms, write_phi, &
       write_phi_kpar, write_rh, write_spec_v_time, write_pzt, &
       write_h_spectrum, write_l_spectrum, write_lh_spectrum, &
       init_single, iky_single, ikx_single, kpar_init, ikx_fixed, iky_fixed, &
       forcing_init, forcing_type, forcing_amp, forcing_index, scale
  ! snyder_electrons, &       
  !       hermite_spectrum_avg_cutoff, &
  ! zero_order_nonlin_flr_only, &
  ! no_nonlin_cross_terms, no_nonlin_dens_cross_term
  ! append_old, nlpm, inlpm, dnlpm, dnlpm_dens, dnlpm_tprp, &
  ! low_b_all, iflr, avail_cpu_time, margin_cpu_time, iso_shear, scan_type, &
  ! scan_number, zero_restart_avg, no_zderiv_covering, no_zderiv, zderiv_loop, &
  ! icovering, no_landau_damping, turn_off_gradients_test, write_netcdf, &
  ! restartfile, no_zonal_nlpm, dorland_qneut, stationary_ions, new_nlpm, &
  ! hammett_nlpm_interference, higher_order_moments, nlpm_zonal_only, &
  ! nlpm_vol_avg, nlpm_abs_sgn, nlpm_kxdep, nlpm_nlps, nlpm_cutoff_avg, &
  ! nlpm_zonal_kx1_only, smagorinsky, dorland_nlpm, dorland_nlpm_phase, &
  ! dorland_phase_complex, dorland_phase_ifac, nlpm_option, dnlpm_max, tau_nlpm

  note = "input"
  
  ! Convert C-string to Fortran character type
  runname = " "
  loop_string: do i=1,256
     if (c_runname(i) == c_null_char) then
        exit loop_string
     else
        runname(i:i) = c_runname(i)
     end if
  end do loop_string

  nml_file = trim(runname)//'.in'

  open(unit=100, file=nml_file, status='old')
  read(unit=100, nml=gx)

  if (nspecies > max_spec) then
     write(*,*) 
     write(*,*) "#####################################"
     write(*,*) 
     write(*,*) "WARNING"
     write(*,*) "nspecies > max_spec = ",max_spec
     write(*,*) "You should recompile or reconsider."
     write(*,*) 
     write(*,*) "#####################################"
     write(*,*) 
  end if
  
  do j = 1, min(max_spec, nspecies)
     call get_indexed_species (j)
     sp(j) % z     = next_sp % z
     sp(j) % mass  = next_sp % mass
     sp(j) % dens  = next_sp % dens
     sp(j) % temp  = next_sp % temp
     sp(j) % tprim = next_sp % tprim
     sp(j) % fprim = next_sp % fprim
     sp(j) % uprim = next_sp % uprim
     sp(j) % vnewk = next_sp % vnewk
     sp(j) % type  = next_sp % type    
  end do
     
  close(unit=100)

  nc_file = trim(runname)//'.nc'

  ! BD: logic to append to a previous run has to go here
  retval = nf90_create (nc_file, NF90_CLOBBER, ncid)

  ! Dimensions known directly from the input file
  retval = nf90_def_dim (ncid, "ri",        ri,             id_ri)
  retval = nf90_def_dim (ncid, "m",         nhermite,       id_M)
  retval = nf90_def_dim (ncid, "l",         nlaguerre,      id_L)
  retval = nf90_def_dim (ncid, "s",         nspecies,       id_nspecies)
!  retval = nf90_def_dim (ncid, "char16",    sixteen,        char16_dim)
  retval = nf90_def_dim (ncid, "time",      NF90_UNLIMITED, id_t_dim)

  ! Header information for the dataset
  file_header = "GX simulation data"
  retval = nf90_put_att (ncid, NF90_GLOBAL, "Title", file_header)
  retval = nf90_def_var (ncid, "ny",        NF90_INT,       id_ny)
  retval = nf90_def_var (ncid, "nx",        NF90_INT,       id_nx)
  retval = nf90_def_var (ncid, "ntheta",    NF90_INT,       id_ntheta)
  retval = nf90_def_var (ncid, "nhermite",  NF90_INT,       id_nm)
  retval = nf90_def_var (ncid, "nlaguerre", NF90_INT,       id_nl)
  retval = nf90_def_var (ncid, "nspecies",  NF90_INT,       id_nspec)
  retval = nf90_def_var (ncid, "dt",        NF90_FLOAT,     id_dt)

  retval = nf90_def_var (ncid, "nperiod",   NF90_INT,       id_nperiod)
  retval = nf90_def_var (ncid, "y0",        NF90_FLOAT,     id_y0)
  retval = nf90_def_var (ncid, "x0",        NF90_FLOAT,     id_x0)
  retval = nf90_def_var (ncid, "nstep",     NF90_INT,       id_nstep)
  retval = nf90_def_var (ncid, "zp",        NF90_INT,       id_zp)
  retval = nf90_def_var (ncid, "jtwist",    NF90_INT,       id_jtwist)
  retval = nf90_def_var (ncid, "i_share",   NF90_INT,       id_i_share)

  !! diagnostics
  retval = nf90_def_var (ncid, "nwrite",    NF90_INT,       id_nwrite)
  retval = nf90_def_var (ncid, "navg",      NF90_INT,       id_navg)
  retval = nf90_def_var (ncid, "nsave",     NF90_INT,       id_nsave)
  retval = nf90_def_var (ncid, "debug",     NF90_INT,       id_debug)    


  retval = nf90_def_var (ncid, "restart",   NF90_INT,       id_restart)    
  retval = nf90_def_var (ncid, "save_for_restart", &
                                            NF90_INT,       id_save_for_restart)    
  retval = nf90_def_var (ncid, "secondary", NF90_INT,       id_secondary)    
  retval = nf90_def_var (ncid, "restart_from_file_dum", &
                                            NF90_INT,       id_restart_from_file_dum)
  retval = nf90_def_var (ncid, "restart_to_file_dum", &
                                            NF90_INT,       id_restart_to_file_dum)
  retval = nf90_def_var (ncid, "eqfix",     NF90_INT,       id_eqfix)    

  retval = nf90_def_var (ncid, "write_omega", &
                                            NF90_INT,       id_write_omega)
  retval = nf90_def_var (ncid, "write_fluxes", &
                                            NF90_INT,       id_write_fluxes)
  retval = nf90_def_var (ncid, "write_moms", &
                                            NF90_INT,       id_write_moms)

  retval = nf90_def_var (ncid, "write_phi", NF90_INT,       id_write_phi)
  retval = nf90_def_var (ncid, "write_phi_kpar", &
                                            NF90_INT,       id_write_phi_kpar)

  retval = nf90_def_var (ncid, "write_h_spectrum", &
                                            NF90_INT,       id_write_h_spectrum) 
  retval = nf90_def_var (ncid, "write_l_spectrum", &
                                            NF90_INT,       id_write_l_spectrum) 
  retval = nf90_def_var (ncid, "write_lh_spectrum", &
                                            NF90_INT,       id_write_lh_spectrum) 
  retval = nf90_def_var (ncid, "write_rh",  NF90_INT,       id_write_rh) 
!  retval = nf90_def_var (ncid, "hermite_spectrum_avg_cutoff", &
!                                            NF90_INT,       id_hspec_cutoff)
  retval = nf90_def_var (ncid, "write_spec_v_time", &
                                            NF90_INT,       id_write_spec_v_time)

  retval = nf90_def_var (ncid, "write_pzt", NF90_INT,       id_write_pzt)
  
  !! numerical parameters
  retval = nf90_def_var (ncid, "cfl",       NF90_FLOAT,     id_cfl)
  retval = nf90_def_var (ncid, "init_amp",  NF90_FLOAT,     id_init_amp)
  retval = nf90_def_var (ncid, "d_hyper",   NF90_FLOAT,     id_d_hyper)
  retval = nf90_def_var (ncid, "nu_hyper",  NF90_FLOAT,     id_nu_hyper)
  retval = nf90_def_var (ncid, "nu_hyper_l", NF90_FLOAT,    id_nu_hyper_l)
  retval = nf90_def_var (ncid, "nu_hyper_m", NF90_FLOAT,    id_nu_hyper_m)
  retval = nf90_def_var (ncid, "p_hyper",   NF90_INT,       id_p_hyper)
  retval = nf90_def_var (ncid, "p_hyper_l", NF90_INT,       id_p_hyper_l)
  retval = nf90_def_var (ncid, "p_hyper_m", NF90_INT,       id_p_hyper_m)

  ! model flags
  retval = nf90_def_var (ncid, "scheme_dum", NF90_INT,      id_scheme_dum)
!  retval = nf90_def_var (ncid, "linear",    NF90_INT,       id_linear)
  retval = nf90_def_var (ncid, "forcing_init", &
                                            NF90_INT,       id_forcing_init)
  retval = nf90_def_var (ncid, "forcing_type_dum", &
                                            NF90_INT,       id_forcing_type_dum)
  retval = nf90_def_var (ncid, "forcing_amp", &
                                            NF90_FLOAT,     id_forcing_amp)
  retval = nf90_def_var (ncid, "forcing_index", &
                                            NF90_INT,       id_forcing_index)
  retval = nf90_def_var (ncid, "phi_ext",   NF90_FLOAT,     id_phi_ext)
  retval = nf90_def_var (ncid, "scale",     NF90_FLOAT,     id_scale)
  retval = nf90_def_var (ncid, "init_field_dum", NF90_INT,  id_init_field_dum)
  retval = nf90_def_var (ncid, "stir_field_dum", NF90_INT,  id_stir_field_dum)
  retval = nf90_def_var (ncid, "init_single", NF90_INT,     id_init_single)
  retval = nf90_def_var (ncid, "ikx_single", NF90_INT,      id_ikx_single)
  retval = nf90_def_var (ncid, "iky_single", NF90_INT,      id_iky_single)
  retval = nf90_def_var (ncid, "ikx_fixed",  NF90_INT,      id_ikx_fixed)
  retval = nf90_def_var (ncid, "iky_fixed",  NF90_INT,      id_iky_fixed)
  retval = nf90_def_var (ncid, "kpar_init", NF90_FLOAT,     id_kpar_init)
  retval = nf90_def_var (ncid, "nonlinear_mode", &   
                                            NF90_INT,       id_nonlinear_mode)

  retval = nf90_def_var (ncid, "boundary_dum", NF90_INT,    id_boundary_dum)
  retval = nf90_def_var (ncid, "iphi00",    NF90_INT,       id_iphi00)
  retval = nf90_def_var (ncid, "source_dum",NF90_INT,       id_source_dum)
  retval = nf90_def_var (ncid, "hyper",     NF90_INT,       id_hyper)
  retval = nf90_def_var (ncid, "hypercollisions", &
                                            NF90_INT,       id_hypercollisions)
  retval = nf90_def_var (ncid, "closure_model_dum", &
                                            NF90_INT,       id_closure_model_dum)
  retval = nf90_def_var (ncid, "smith_par_q", &
                                            NF90_INT,       id_smith_par_q)
  retval = nf90_def_var (ncid, "smith_perp_q", &
                                            NF90_INT,       id_smith_perp_q)

  ! geometry
  retval = nf90_def_var (ncid, "igeo",      NF90_INT,       id_igeo)
  retval = nf90_def_var (ncid, "slab",      NF90_INT,       id_slab)
  retval = nf90_def_var (ncid, "const_curv", NF90_INT,      id_const_curv)
  retval = nf90_def_var (ncid, "geofile_dum",NF90_INT,      id_geofile_dum)

  retval = nf90_def_var (ncid, "drhodpsi",  NF90_FLOAT,     id_drhodpsi)    
  retval = nf90_def_var (ncid, "kxfac",     NF90_FLOAT,     id_kxfac)       
  retval = nf90_def_var (ncid, "Rmaj",      NF90_FLOAT,     id_Rmaj)        
  retval = nf90_def_var (ncid, "shift",     NF90_FLOAT,     id_shift)       
  retval = nf90_def_var (ncid, "eps",       NF90_FLOAT,     id_eps)         
  retval = nf90_def_var (ncid, "rhoc",      NF90_FLOAT,     id_rhoc)        
  retval = nf90_def_var (ncid, "q",         NF90_FLOAT,     id_qinp)         
  retval = nf90_def_var (ncid, "shat",      NF90_FLOAT,     id_shat)        
  retval = nf90_def_var (ncid, "kappa",      NF90_FLOAT,    id_akappa)       
  retval = nf90_def_var (ncid, "kappa_prime", NF90_FLOAT,   id_akappri) 
  retval = nf90_def_var (ncid, "tri",       NF90_FLOAT,     id_tri)         
  retval = nf90_def_var (ncid, "tri_prime", NF90_FLOAT,     id_tripri)   

  retval = nf90_def_var (ncid, "beta",      NF90_FLOAT,     id_beta)
  retval = nf90_def_var (ncid, "beta_prime_input", &
                                            NF90_FLOAT,     id_beta_prime_input)
  retval = nf90_def_var (ncid, "s_hat_input", NF90_FLOAT,   id_s_hat_input)

  ! species data...just one species for now
  retval = nf90_def_var (ncid, "spec_type_dum",NF90_INT,    id_spec_type_dum)
  retval = nf90_def_var (ncid, "z",         NF90_FLOAT,     id_spec_z)     
  retval = nf90_def_var (ncid, "m",         NF90_FLOAT,     id_spec_mass)  
  retval = nf90_def_var (ncid, "n0",        NF90_FLOAT,     id_spec_dens)  
  retval = nf90_def_var (ncid, "n0_prime",  NF90_FLOAT,     id_spec_fprim) 
  retval = nf90_def_var (ncid, "u0_prime",  NF90_FLOAT,     id_spec_uprim) 
  retval = nf90_def_var (ncid, "T0",        NF90_FLOAT,     id_spec_temp)  
  retval = nf90_def_var (ncid, "T0_prime",  NF90_FLOAT,     id_spec_tprim) 
  retval = nf90_def_var (ncid, "nu",        NF90_FLOAT,     id_spec_vnewk)  
  retval = nf90_def_var (ncid, "tite",      NF90_FLOAT,     id_tite)       

  retval = nf90_def_var (ncid, "fphi",      NF90_FLOAT,     id_fphi)
  retval = nf90_def_var (ncid, "fapar",     NF90_FLOAT,     id_fapar)
  retval = nf90_def_var (ncid, "fbpar",     NF90_FLOAT,     id_fbpar)

!  retval = nf90_def_var (ncid, "snyder_electrons", &
!                                            NF90_INT,       id_snyder_electrons)

  retval = nf90_put_att (ncid, id_boundary_dum,      "value", trim(boundary))
  retval = nf90_put_att (ncid, id_closure_model_dum, "value", trim(closure_model))
  retval = nf90_put_att (ncid, id_scheme_dum,        "value", trim(scheme))
  retval = nf90_put_att (ncid, id_geofile_dum,       "value", trim(geofile))
  retval = nf90_put_att (ncid, id_source_dum,        "value", trim(source))
  retval = nf90_put_att (ncid, id_init_field_dum,    "value", trim(init_field))
  retval = nf90_put_att (ncid, id_stir_field_dum,    "value", trim(stir_field))
  retval = nf90_put_att (ncid, id_forcing_type_dum,  "value", trim(forcing_type))
  retval = nf90_put_att (ncid, id_spec_type_dum,     "value", trim(sp(1) % type))
  retval = nf90_put_att (ncid, id_restart_from_file_dum, "value", trim(restart_from_file))
  retval = nf90_put_att (ncid, id_restart_to_file_dum,   "value", trim(restart_to_file))

  retval = nf90_def_var (ncid, 'code_info', NF90_INT, id_code)

  datestamp(:) = ' '
  timestamp(:) = ' '
  timezone(:) = ' '
  call date_and_time (datestamp, timestamp, timezone)

  ci = 'c1'
  retval = nf90_put_att (ncid, id_code, trim(ci), 'Date: '//trim(datestamp))

  ci = 'c2'
  retval = nf90_put_att (ncid, id_code, trim(ci), 'Time: '//trim(timestamp)//' '//trim(timezone)//' ')

  ci = 'c3'
  retval = nf90_put_att (ncid, id_code, trim(ci), 'NetCDF version '//trim(nf90_inq_libvers()))

  ci = 'c4'
  retval = nf90_put_att (ncid, id_code, trim(ci), 'This code assumes vt == sqrt(T/m) ')

  retval = nf90_enddef (ncid)

  j=debug;             retval = nf90_put_var (ncid, id_debug, j)
  j=restart;           retval = nf90_put_var (ncid, id_restart, j)
  j=save_for_restart;  retval = nf90_put_var (ncid, id_save_for_restart, j)
  j=eqfix;             retval = nf90_put_var (ncid, id_eqfix, j)
  j=secondary;         retval = nf90_put_var (ncid, id_secondary, j)
  j=nonlinear_mode;    retval = nf90_put_var (ncid, id_nonlinear_mode, j)
  j=slab;              retval = nf90_put_var (ncid, id_slab, j)
  j=const_curv;        retval = nf90_put_var (ncid, id_const_curv, j)
  j=hyper;             retval = nf90_put_var (ncid, id_hyper, j)
  j=hypercollisions;   retval = nf90_put_var (ncid, id_hypercollisions, j)
  j=write_omega;       retval = nf90_put_var (ncid, id_write_omega, j)
  j=write_fluxes;      retval = nf90_put_var (ncid, id_write_fluxes, j)
  j=write_moms;        retval = nf90_put_var (ncid, id_write_moms, j)
  j=write_phi;         retval = nf90_put_var (ncid, id_write_phi, j)
  j=write_phi_kpar;    retval = nf90_put_var (ncid, id_write_phi_kpar, j)
  j=write_h_spectrum;  retval = nf90_put_var (ncid, id_write_h_spectrum, j)
  j=write_l_spectrum;  retval = nf90_put_var (ncid, id_write_l_spectrum, j)
  j=write_lh_spectrum; retval = nf90_put_var (ncid, id_write_lh_spectrum, j)
  j=write_rh;          retval = nf90_put_var (ncid, id_write_rh, j)
  j=write_spec_v_time; retval = nf90_put_var (ncid, id_write_spec_v_time, j)
  j=init_single;       retval = nf90_put_var (ncid, id_init_single, j)
!  j=snyder_electrons;  retval = nf90_put_var (ncid, id_synder_electrons, j)
  j=forcing_init;      retval = nf90_put_var (ncid, id_forcing_init, j)
  j=write_pzt;         retval = nf90_put_var (ncid, id_write_pzt, j)

  retval = nf90_put_var (ncid, id_ny,               ny)
  retval = nf90_put_var (ncid, id_nx,               nx)
  retval = nf90_put_var (ncid, id_nm,               nhermite)
  retval = nf90_put_var (ncid, id_nl,               nlaguerre)
  retval = nf90_put_var (ncid, id_ntheta,           ntheta)
  retval = nf90_put_var (ncid, id_nperiod,          nperiod)
  retval = nf90_put_var (ncid, id_nspec,            nspecies)
  retval = nf90_put_var (ncid, id_smith_par_q,      smith_par_q)
  retval = nf90_put_var (ncid, id_smith_perp_q,     smith_perp_q)
  retval = nf90_put_var (ncid, id_nstep,            nstep)
  retval = nf90_put_var (ncid, id_nwrite,           nwrite)
  retval = nf90_put_var (ncid, id_navg,             navg)
  retval = nf90_put_var (ncid, id_nsave,            nsave)
  retval = nf90_put_var (ncid, id_jtwist,           jtwist)
  retval = nf90_put_var (ncid, id_iphi00,           iphi00)
  retval = nf90_put_var (ncid, id_igeo,             igeo)
  retval = nf90_put_var (ncid, id_p_hyper,          p_hyper)
  retval = nf90_put_var (ncid, id_p_hyper_l,        p_hyper_l)
  retval = nf90_put_var (ncid, id_p_hyper_m,        p_hyper_m)
!  retval = nf90_put_var (ncid, id_hspec_cutoff,     hermite_spectrum_avg_cutoff)
  retval = nf90_put_var (ncid, id_ikx_single,       ikx_single)
  retval = nf90_put_var (ncid, id_iky_single,       iky_single)
  retval = nf90_put_var (ncid, id_ikx_fixed,        ikx_fixed)
  retval = nf90_put_var (ncid, id_iky_fixed,        iky_fixed)
  retval = nf90_put_var (ncid, id_forcing_index,    forcing_index)
  
  retval = nf90_put_var (ncid, id_x0,               x0)
  retval = nf90_put_var (ncid, id_y0,               y0)
  retval = nf90_put_var (ncid, id_dt,               dt)
  retval = nf90_put_var (ncid, id_zp,               zp)
  retval = nf90_put_var (ncid, id_tite,             tite)
  retval = nf90_put_var (ncid, id_fphi,             fphi)
  retval = nf90_put_var (ncid, id_fapar,            fapar)
  retval = nf90_put_var (ncid, id_fbpar,            fbpar)
  retval = nf90_put_var (ncid, id_beta,             beta)
  retval = nf90_put_var (ncid, id_cfl,              cfl)
  retval = nf90_put_var (ncid, id_rhoc,             rhoc)
  retval = nf90_put_var (ncid, id_eps,              eps)
  retval = nf90_put_var (ncid, id_shat,             shat)
  retval = nf90_put_var (ncid, id_qinp,             qinp)
  retval = nf90_put_var (ncid, id_shift,            shift)
  retval = nf90_put_var (ncid, id_akappa,           akappa)
  retval = nf90_put_var (ncid, id_akappri,          akappri)
  retval = nf90_put_var (ncid, id_tri,              tri)
  retval = nf90_put_var (ncid, id_tripri,           tripri)
  retval = nf90_put_var (ncid, id_Rmaj,             Rmaj)
  retval = nf90_put_var (ncid, id_beta_prime_input, beta_prime_input)
  retval = nf90_put_var (ncid, id_s_hat_input,      s_hat_input)
  retval = nf90_put_var (ncid, id_drhodpsi,         drhodpsi)
  retval = nf90_put_var (ncid, id_kxfac,            kxfac)
  retval = nf90_put_var (ncid, id_phi_ext,          phi_ext)
  retval = nf90_put_var (ncid, id_nu_hyper,         nu_hyper)
  retval = nf90_put_var (ncid, id_d_hyper,          d_hyper)
  retval = nf90_put_var (ncid, id_nu_hyper_l,       nu_hyper_l)
  retval = nf90_put_var (ncid, id_nu_hyper_m,       nu_hyper_m)
  retval = nf90_put_var (ncid, id_init_amp,         init_amp)
  retval = nf90_put_var (ncid, id_kpar_init,        kpar_init)
  retval = nf90_put_var (ncid, id_forcing_amp,      forcing_amp)
  retval = nf90_put_var (ncid, id_scale,            scale)
  retval = nf90_put_var (ncid, id_i_share,          i_share)

  retval = nf90_put_var (ncid, id_boundary_dum,     boundary_dum)
  retval = nf90_put_var (ncid, id_closure_model_dum,closure_model_dum)  
  retval = nf90_put_var (ncid, id_scheme_dum,       scheme_dum)
  retval = nf90_put_var (ncid, id_geofile_dum,      geofile_dum)
  retval = nf90_put_var (ncid, id_source_dum,       source_dum)
  retval = nf90_put_var (ncid, id_init_field_dum,   init_field_dum)
  retval = nf90_put_var (ncid, id_stir_field_dum,   stir_field_dum)
  retval = nf90_put_var (ncid, id_forcing_type_dum, forcing_type_dum)
  retval = nf90_put_var (ncid, id_restart_from_file_dum, restart_from_file_dum)
  retval = nf90_put_var (ncid, id_restart_to_file_dum,   restart_to_file_dum)

  ! species data...just one species for now
  retval = nf90_put_var (ncid, id_spec_type_dum,    spec_type_dum)
  retval = nf90_put_var (ncid, id_spec_z,           sp(1) % z)
  retval = nf90_put_var (ncid, id_spec_mass,        sp(1) % mass)
  retval = nf90_put_var (ncid, id_spec_dens,        sp(1) % dens)
  retval = nf90_put_var (ncid, id_spec_fprim,       sp(1) % fprim)
  retval = nf90_put_var (ncid, id_spec_uprim,       sp(1) % uprim)
  retval = nf90_put_var (ncid, id_spec_temp,        sp(1) % temp)
  retval = nf90_put_var (ncid, id_spec_tprim,       sp(1) % tprim)    
  retval = nf90_put_var (ncid, id_spec_vnewk,       sp(1) % vnewk)

  retval = nf90_close (ncid)
  
contains
  
  subroutine get_indexed_species (index)

    implicit none
    integer, intent (in) :: index
!    type (plasma), intent (inout) :: next_sp
    real :: z, mass, dens, vnewk, temp, tprim, fprim, uprim
    character (len=8) :: type

    namelist /species_parameters_1/z, mass, dens, vnewk, &
         temp, tprim, fprim, uprim, type
    namelist /species_parameters_2/z, mass, dens, vnewk, &
         temp, tprim, fprim, uprim, type
    namelist /species_parameters_3/z, mass, dens, vnewk, &
         temp, tprim, fprim, uprim, type
    
    select case (index)
    case (1)
       read(unit=100, nml=species_parameters_1)
       next_sp % z     = z
       next_sp % mass  = mass
       next_sp % dens  = dens
       next_sp % temp  = temp
       next_sp % tprim = tprim
       next_sp % fprim = fprim
       next_sp % uprim = uprim
       next_sp % vnewk = vnewk
       next_sp % type  = type    
    case (2)
       read(unit=100, nml=species_parameters_2)
       next_sp % z     = z
       next_sp % mass  = mass
       next_sp % dens  = dens
       next_sp % temp  = temp
       next_sp % tprim = tprim
       next_sp % fprim = fprim
       next_sp % uprim = uprim
       next_sp % vnewk = vnewk
       next_sp % type  = type    
    case (3)
       read(unit=100, nml=species_parameters_3)
       next_sp % z     = z
       next_sp % mass  = mass
       next_sp % dens  = dens
       next_sp % temp  = temp
       next_sp % tprim = tprim
       next_sp % fprim = fprim
       next_sp % uprim = uprim
       next_sp % vnewk = vnewk
       next_sp % type  = type    
    end select
  end subroutine get_indexed_species
end subroutine read_nml

