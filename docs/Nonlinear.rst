.. _quicknl:

Running your first nonlinear simulation
+++++++++++++++++++++++++++++++++++++++

In this tutorial we set up a nonlinear ITG turbulence calculation using a circular Miller geometry with Cyclone-base-case-like parameters and adiabatic electrons.

.. contents::

Input file
----------

The :doc:`input file <inputFiles/nl_miller_adiabatic_electrons>` for this case is included in the GX repository in ``benchmarks/nonlinear/cyclone/cyclone_miller_adiabatic_electrons.in``.

The input file for this nonlinear case is similar to the :ref:`linear <quicklin>` example, but we will highlight key differences below. For more details about input parameters, see :ref:`input_file`.

Dimensions
==========

.. code-block:: toml

  [Dimensions]
   ntheta = 24            # number of points along field line (theta) per 2pi segment    
   nperiod = 1            # number of 2pi segments along field line is 2*nperiod-1
   ny = 64                # number of real-space grid-points in y
   nx = 192               # number of real-space grid-points in x
  
   nhermite = 8           # number of hermite moments (v_parallel resolution)
   nlaguerre = 4          # number of laguerre moments (mu B resolution)
   nspecies = 1           # number of evolved kinetic species (adiabatic electrons don't count towards nspecies)

Unlike in the :ref:`linear <lindims>` calculation where we set ``nkx`` and ``nky``, for a nonlinear calculation it is recommended to set the perpendicular resolution using ``nx`` and ``ny``. These are the number of real-space grid-points used in the radial and binormal coordinates when computing the nonlinear term pseudo-spectrally. Because of the need to prevent aliasing of Fourier modes, the number of evolved Fourier modes is less: ``nkx = 1 + 2*(nx-1)/3 = 64`` and ``nky = 1 + (ny-1)/3 = 22``. 

Further, it is recommended to set ``nperiod = 1`` for nonlinear calculations; when ``nx>1``, extension of modes along the field line is captured by twist-shift boundary conditions that link together several modes on an extended :math:`\theta` domain. 

We have also lowered the velocity resolution for this nonlinear calculation compared to the linear one. This is done mainly so that this example runs quickly. Nevertheless, the calculation is still quite accurate even at this fairly low velocity resolution due to GX's choice of velocity-space coordinates and spectral approach.

Domain
======

.. code-block:: toml

  [Domain]
   y0 = 28.2              # controls box length in y (in units of rho_ref) and minimum ky, so that ky_min*rho_ref = 1/y0 
   boundary = "linked"    # use twist-shift boundary conditions along field line
  
The ``[Domain]`` group here is similar to the one from the :ref:`linear <lindom>` calculation. Here, setting the binormal box length with ``y0=28.2`` means that the minimum binormal wavenumber will be :math:`k_{y\, \mathrm{min}} = 0.0355`, and the maximum binormal wavenumber will be :math:`k_{y\,\mathrm{max}} = \texttt{nky/y0} = 0.78`. What about the radial box length? By default (when there is finite magnetic shear), the radial box length is set to be approximately the same as the binormal box length. It usually cannot be exactly equal because the radial box length must be specially quantized so that the twist-shift (``"linked"``) boundary condition can be applied appropriately.
