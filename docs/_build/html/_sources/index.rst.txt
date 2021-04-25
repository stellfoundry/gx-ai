.. GX documentation master file, created by
   sphinx-quickstart on Wed Apr 21 21:02:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for the GX code
==============================

  
	     
GX is a code for solving the nonlinear gyrokinetic equations for
plasma turbulence within a Fourier-Hermite-Laguerre framework.
This allows GX to smoothly interpolate between coarse
gyrofluid-like resolutions and finer conventional gyrokinetic
resolutions.

To get started, you will need an NVIDIA GPU, and the gsl (cpu), NetCDF
(cpu), and CUDA (gpu) libraries, including cuTensor.

GX is currently supported on Traverse at Princeton, where you can
simply module load cuda and set the GK_SYSTEM environment variable
with

$ export GK_SYSTEM=traverse

On other systems, you will need to make a custom
Makefiles/Makefile.[SYSTEM] with the paths to the CUDA libraries (use
Makefile.maverick as an example) and set the GK_SYSTEM environment
variable to [SYSTEM].

After taking care of the system Makefile, to build GX simply execute
make in the main directory. This will build the 'gx' executable.

To run GX, execute

$ ./gx [path_to_inputfile]

A sample input file is provided in ______. 


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Overview
   Geometry
   Inputs
   Outputs
   Numerics
   
