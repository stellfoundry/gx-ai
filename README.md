GX
=====

GX is a code for solving the nonlinear gyrokinetic equation for plasma turbulence using a spectral Hermite-Laguerre velocity discretization.
This allows GX to smoothly interpolate between coarse gyrofluid-like resolutions and finer conventional gyrokinetic resolutions.

To get started, you will need an NVIDIA GPU and the gsl (cpu) and CUDA (gpu) libraries.

GX is currently supported on Traverse at Princeton, where you can simply module load cuda and set the GK_SYSTEM environment variable with 

$ export GK_SYSTEM=traverse

On other systems, you will need to make a custom Makefiles/Makefile.[SYSTEM] with the paths to the CUDA libraries 
(use Makefile.maverick as an example) and set the GK_SYSTEM environment variable to [SYSTEM].

After taking care of the system Makefile, to build GX simply execute make in the main directory. This will build the 'gx' executable.

To run GX, execute

$ ./gx [path_to_inputfile]

A sample linear input file is provided in inputs/cyclone_linear/cyc_lin.in

