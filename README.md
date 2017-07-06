GRYFX++
=====

Gryfx++ is a code for solving the nonlinear gyrokinetic equation for plasma turbulence using a spectral Hermite-Laguerre velocity discretization.
This allows Gryfx++ to smoothly interpolate between coarse gyrofluid-like resolutions and finer conventional gyrokinetic resolutions.

To get started, you will need an NVIDIA GPU and the CUDA libraries.

Gryfx++ is currently supported on Maverick at TACC, where you can simply module load cuda and set the GK_SYSTEM environment variable with 
$ export GK_SYSTEM=maverick. 
On other systems, you will need to make a custom Makefiles/Makefile.[SYSTEM] with the paths to the CUDA libraries 
(use Makefile.maverick as an example) and set the GK_SYSTEM environment variable to [SYSTEM].

After taking care of the system Makefile, to build Gryfx++ simply execute make in the main directory. This will build the 'gryfx++' executable.

To run Gryfx++, execute
$ ./gryfx++ [path_to_inputfile]

A sample linear input file is provided in inputs/cyclone_linear.in.

