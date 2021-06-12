Inputs
======

GX is currently set up to be run from input files. The main input file has a user-supplied name,
such as ``a01`` or ``scan/a01``, to which the suffix **.in** should be appended. In these examples, 
the input file names are ``a01.in`` and ``scan/a01.in``. In the first case, the input file is in
the current directory. In the second case, the input file is in a directory called ``scan``,
which itself is located in the current directory.

Important: If you are using an input file in another directory, and if you are also referring to additional
files (perhaps for a restart, perhaps to specify geometric information, etc) then the file names
for the additional files must also include the longer path.


The input file
--------------
*There are two standards for the input file. One is deprecated and is not documented here.
If you are exploring src/parameters.cu and find yourself confused by the logic there, you are
almost certainly looking at obscure coding that supports the old standard. That obscure coding
is itself deprecated.*

The main GX input file is parsed with `toml <https://github.com/ToruNiina/toml11>`_.

The toml standard allows one to group related input variables. There are multiple
ways to specify such groupings. Since there are sample input files provided
with GX, here we will pick one particular syntactical pattern. Users are free
to use anything that is compatible with the `toml <https://github.com/ToruNiina/toml11>`_ standard.

There are two input variables that are not part of any group:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - *none*
     - debug
     - If true, print debug information to standard out
     - **false**
   * - *none*
     - repeat 
     - If true, repeat a previous run, using the information in the
       output file from that run. **Presently not available**
     - **false**

The grid sizes are specied in the **Dimensions** group:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Dimensions**
     - nx
     - The number of grid points used in the *x* direction.
     - **4**
   * - **Dimensions**
     - ny
     - The number of grid points used in the *y* direction.
     - **32**
   * - **Dimensions**
     - ntheta
     - The number of grid points used in the *z* direction.
     - **32**
   * - **Dimensions**
     - nhermite
     - The number of Hermite basis functions used.
     - **4**
   * - **Dimensions**
     - nlaguerre
     - The number of Laguerre basis functions used.
     - **2**
   * - **Dimensions**
     - nspecies
     - The number of kinetic species used.
     - **1**
   * - **Dimensions**
     - nperiod
     - The number of poloidal turns used. 
     - **1**

The physical extent of the simulation domain is specied in the **Domain** group:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Domain**
     - x0
     - The extent of the radial domain is 2 pi x0. 
     - **10.0**
   * - **Domain**
     - y0
     - The extent of the binormal domain is 2 pi y0. 
     - **10.0**
   * - **Domain**
     - jtwist
     - The twist-and-shift boundary condition is controlled by jtwist. Typically,
       you should set jtwist to be an integer close to 2 pi / s_hat. 
     - **-1**
   * - **Domain**
     - boundary
     - Two options are possible: **"periodic"** or **"linked"**
     - **"linked"**
   * - **Domain**
     - ExBshear
     - **Not yet implemented.** Set to **true** to include equilibrium ExB shear. 
     - **false**
   * - **Domain**
     - g_exb
     - **Not yet implemented.** Sets the equilibrium ExB shearing rate. 
     - **0.0**

The length of simulated time and the timestep are set in the **Time** group:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Time**
     - dt
     - The maximum timestep allowed.
     - **0.05**
   * - **Time**
     - nstep
     - The number of timesteps to take. 
     - **10000**
   * - **Time**
     - nwrite
     - Write time-dependent information every nwrite timesteps. 
     - **1000**
   * - **Time**
     - navg
     - **Not yet implemented.** Average time-dependent information over navg timesteps.
     - **10**
   * - **Time**
     - nsave
     - **Not yet implemented.** Write a restart file every nsave timesteps. 
     - **2000000**

The **Controls** group contains switches and variables that determine both the physics model
and various numerical parameters.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Controls**
     - nonlinear_mode
     - Set to true to include nonlinear terms in the equations. 
     - **false**
   * - **Controls**
     - scheme
     - This string variable chooses the time-stepping scheme to be used. 
     - **"sspx3"**
   * - **Controls**
     - stages
     - The number of Runge-Kutta stages to be used for certain time-stepping schemes. Not relevant
       for most choices of **scheme**. 
     - **10**    
   * - **Controls**
     - cfl
     - For nonlinear runs, the maximum timestep allowed is proportional to **cfl**.
     - **1.0**    
   * - **Controls**
     - init_field
     - The initial perturbation is applied to this component of the distribution function.
     - **"density"**    
   * - **Controls**
     - init_amp
     - The initial perturbation has this amplitude. 
     - **1.0e-5**    
   * - **Controls**
     - kpar_init
     - The initial perturbation has this parallel wavenumber.
     - **0.0**
   * - **Controls**
     - closure_model
     - This string variable determines which (if any) closure model to use.
     - **"none"**
   * - **Controls**
     - fphi
     - Set fphi to 1.0 to include Phi perturbations in the calculation. 
     - **1.0** 
   * - **Controls**
     - fapar
     - **Not yet implemented.** Set fapar to 1.0 to include perturbations of the
       parallel component of the vector potential in the calculation. 
     - **0.0**
   * - **Controls**
     - fbpar
     - **Not yet implemented.** Set fbpar to 1.0 to include perturbations of the
       magnetic field strength in the calculation. 
     - **0.0**
   * - **Controls**
     - HB_hyper
     - If true, use the Hammett-Belli hyperdiffusivity model 
     - **false**
   * - **Controls**
     - D_HB
     - If HB_hyper is true, sets the strength of the H-B hyperdiffusivity model.
     - **1.0**
   * - **Controls**
     - w_osc
     - If HB_hyper is true, sets the frequency parameter in the H-B model.
     - **0.0**
   * - **Controls**
     - p_HB
     - If HB_hyper is true, sets the exponent for the H-B model,
       where 2 corresponds to the fourth power of k.
     - **2**

       
To continue a previous run, use the  **Restart** group:

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Restart**
     - restart
     - Set to true to continue from a previous run.
     - **false**
   * - **Restart**
     - save_for_restart
     - Set to true to write information needed for a future restart. File is written at the end of the run. 
     - **true**
   * - **Restart**
     - restart_to_file
     - Filename to use for restart information for the present run.       
     - **"newsave.nc"**
   * - **Restart**
     - restart_from_file
     - Filename for a file written from a previous run, to be used as the basis for continuation (restart).
     - **"oldsave.nc"**
   * - **Restart**
     - scale
     - Multiply all variables in the restart variable by a factor of **scale**. 
     - **1.0**

The **Diagnostics** group controls the kinds of diagnostic information that is produced.
Diagnostic information is written in NetCDF format, to a file that uses the input file name
(without the **.in** suffix) with the **.nc** suffix appended.

.. list-table::
   :widths: 20 20 50 10
   :width: 100
   :header-rows: 1

   * - Group
     - Variable
     - Description
     - Default
   * - **Diagnostics**
     - omega
     - Write instantaneous estimates of the complex frequency for each Fourier component of the electrostatic potential.
     - **false**
   * - **Diagnostics**
     - free_energy
     - Write the total free energy (integrated over the phase-space domain and summed over species) as a function of time. 
     - **true**
   * - **Diagnostics**
     - fluxes
     - Write the turbulent fluxes for each species. 
     - **false**
   * - **Diagnostics**
     - all_zonal_scalars
     - Write quantities such as the RMS value of the zonal component of the ExB velocity as function of time.
       This is a shortcut for turning on all such writes, instead of specifying them individually (below).
     - **false**
   * - **Diagnostics**
     - avg_zvE
     - Write the RMS value of the zonal component of ExB velocity as a function time.
     - **false**
   * - **Diagnostics**
     - avg_zkvE
     - Write the RMS value of the zonal component of the shear of the ExB velocity as a function time.
     - **false**
   * - **Diagnostics**
     - avg_zkden
     - Write the RMS value of the zonal component of the guiding center
       radial density gradient as a function of time.
     - **false**
   * - **Diagnostics**
     - avg_zkUpar
     - Write the RMS value of the zonal component of the guiding center
       parallel velocity as a function of time.
     - **false**
   * - **Diagnostics**
     - avg_zkTpar
     - Write the RMS value of the zonal component of the guiding center
       parallel temperature as a function of time.
     - **false**
   * - **Diagnostics**
     - avg_zkTperp
     - Write the RMS value of the zonal component of the guiding center
       perpendicular temperature as a function of time.
     - **false**
   * - **Diagnostics**
     - avg_zkqpar
     - Write the RMS value of the zonal component of the guiding center
       parallel-parallel heat flux as a function of time.
     - **false**
   * - **Diagnostics**
     - all_zonal
     - Write quantities such as the zonal component of the ExB velocity as function of *x* and time.
       This is a shortcut for turning on all such writes, instead of specifying them individually (below).
     - **false**
   * - **Diagnostics**
     - vE
     - Write the zonal component of ExB velocity as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - kvE
     - Write the zonal component of the shear of the ExB velocity as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - kden
     - Write the zonal component of the radial gradient of the guiding center
       density as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - kUpar
     - Write the zonal component of the radial gradient of the guiding center
       parallel velocity as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - kTpar
     - Write the zonal component of the radial gradient of the guiding center
       parallel temperature as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - kTperp
     - Write the zonal component of the radial gradient of the guiding center
       perpendicular temperature as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - kqpar
     - Write the zonal component of the radial gradient of the guiding center
       parallel-parallel heat flux as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - all_non_zonal
     - Write quantities such as the non-zonal *y*-component of the ExB velocity as function of *x* and time.
       This is a shortcut for turning on all such writes, instead of specifying them individually (below).
     - **false**
   * - **Diagnostics**
     - xyvE
     - Write the non-zonal, *y*-component of the ExB velocity as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - xykvE
     - Write the non-zonal part of the shear of the *y*-component of the ExB velocity as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - xyden
     - Write the non-zonal component of the guiding center density as a function of *x* and time.       
     - **false**
   * - **Diagnostics**
     - xyUpar
     - Write the non-zonal component of the guiding center parallel velocity as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - xyTpar
     - Write the non-zonal component of the guiding center parallel temperature as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - xyTperp
     - Write the non-zonal component of the guiding center perpendicular temperature as a function of *x* and time.
     - **false**
   * - **Diagnostics**
     - xyqpar
     - Write the non-zonal component of the guiding center parallel-parallel heat flux as a function of *x* and time.
     - **false**

Auxiliary files
---------------
