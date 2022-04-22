# GX

## Introduction

GX is a code for solving the nonlinear gyrokinetic equation for low-frequency turbulence in magnetized plasmas. 
A unique feature of GX is the use of a spectral Hermite-Laguerre velocity discretization. 
This allows GX to smoothly interpolate between coarse gyrofluid-like resolutions and finer conventional gyrokinetic resolutions.

Another unique feature of GX is that it is a GPU-native code, designed and optimized in CUDA/C++. 
This means you will need access to an NVIDIA GPU to run GX. 

User documentation for the code is available at https://gx.rtfd.io (a work in progress).

GX is currently under rapid development, resulting in quickly-changing functionality and capabilities. 
A number of planned improvements to the code (such as multi-GPU capability) are listed in the issue tracker on BitBucket (https://bitbucket.org/gyrokinetics/gx/issues).

## Dependencies

In addition to the NVCC compiler, the following external dependencies are required to build GX:

- CUDA libraries (all of these are included with the CUDA toolkit): 
	- CUDA Runtime
	- cuFFT
	- cuBLAS
	- cuSOLVER
	- cuTENSOR
	- cuLIBOS
- NetCDF
- HDF5
- GSL

## Building the code

GX is currently supported on several systems, enabling a relatively simple build process.
To see if your system (e.g. `traverse` or `stellar`) is supported, check to see if there is a Makefile corresponding to your system in the `Makefiles` directory.

If you are on a supported system (we'll use `traverse` as an example):

1. Set the `GK_SYSTEM` environment variable to the name of the system. For `traverse`, this can be accomplished with (assuming `bash`)
```
$ export GK_SYSTEM='traverse'
```
2. Load required modules for the system, if available. The necessary commands can be found in the comments at the top of the `Makefiles/Makefile.[GK_SYSTEM]` file.
Typically it is easiest to add these module load commands to your `.bashrc` file (or equivalent) so that the necessary modules (which are needed at compile-time and at run-time)
are always automatically loaded.
3. Build GX by simply executing `make` (or `make -j` to build in parallel) in the main directory:
```
$ make [-j]
```
This will create the `gx` executable in the main directory.

If you are not on a supported system, you will need to make a custom `Makefiles/Makefile.[GK_SYSTEM]`
with the paths to the necessary libraries; please use `Makefile.traverse` as a template. 

## Running GX

To run GX using an input file named `run_name.in`, execute
```
$ [path/to/]gx [path/to/]run_name.in
```

Sample input files for standard benchmark cases are provided in the `benchmarks` directory.
For example, to run the linear adiabatic-electron cyclone base case benchmark, 
navigate into the `benchmarks/linear/cyclone_ae/` directory, and execute
```
$ ../../gx cyclone_ae.in
```
Diagnostic output will be printed to the screen, and also to the `cyclone_ae.nc` NetCDF output file.

## Citing GX

If you use GX in your work, please cite the following papers:

N. R. Mandell, W. D. Dorland, and M. Landreman. 2018. "Laguerre-Hermite pseudo-spectral velocity formulation of gyrokinetics". J. Plasma Phys. **84**, 9058[]()40108. https://doi.org/10.1017/S0022377818000041

The GX code paper is in progress, please stay tuned!

N. R. Mandell, et al. (in prep). "GX: a GPU-native gyrokinetic turbulence code for tokamaks and stellarators". 

## License

Copyright (c) 2011-2022 Noah R. Mandell, William D. Dorland, and the GX team.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
