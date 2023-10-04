#!/usr/bin/env python
"""
This a Pythonized geometry module to read tokamak and stellarator equilibria from a VMEC file and caculate the geometric coefficients needed for a GX/GS2 run. Additionally, this module can vary the pressure and iota gradients self-consistently (while respecting MHD force balance) according to the work by Greene and Chance + Hegna and Nakajima and recalculate the geometry coefficients.

Dependencies:
netcdf4, pip install netcdf4
booz_xform, pip install booz_xform

For axisymmetric equilibria
python gx_geo_vmec.py <vmec_filename(with .nc)> 1 <desired output name>

For 3D equilibria
python gx_geo_vmec.py <vmec_filename(with .nc)> 0 <desired output name>

A portion of this script is based on Matt Landreman's vmec_geometry module for the SIMSOPT framework.
For axisymmetric equilibria, make sure that ntor > 1 in the VMEC wout file.
"""

import numpy as np

from scipy.optimize import newton
from scipy.interpolate import (
    interp1d,
    InterpolatedUnivariateSpline,
)
from scipy.integrate import cumulative_trapezoid as ctrap
from scipy.integrate import simpson as simps

from netCDF4 import Dataset as ds
import booz_xform as bxform

from matplotlib import pyplot as plt

import pdb
import sys

vmec_fname = sys.argv[1]
isaxisym = int(eval(sys.argv[2]))
eikfile = sys.argv[3]

mu_0 = 4 * np.pi * (1.0e-7)


################################################################################################
##############-----------------------HELPER FUNCTIONS---------------------------################
################################################################################################


def nperiod_set(arr, nperiod, extend=True, brr=None):
    """
    Contract or extend a large array to a smaller one.
    Truncates (or extends) an array to a smaller one. This function gives us the ability to truncate a
    variable such as |B| to theta \in [-\pi, pi].
    Inputs
    ------
    arr: numpy array
    Input array of the dependent variable
    brr: numpy array
    Input array of the independent variable
    extend: boolean
    Whether to extend instead of contract a large array

    Returns
    -------
    A truncated or extended array arr
    """
    if extend is True and nperiod > 1:
        arr_temp0 = arr - (arr[0] + (2 * nperiod - 1) * np.pi)
        arr_temp1 = arr_temp0
        for i in range(nperiod):
            arr_temp1 = np.concatenate((arr_temp1, arr_temp0[1:] + 2 * np.pi * (i + 1)))

        arr = arr_temp1
    elif brr is None:  # contract the theta array
        eps = 1e-11
        arr_temp0 = arr[arr <= (2 * nperiod - 1) * np.pi + eps]
        arr_temp0 = arr_temp0[arr_temp0 >= -(2 * nperiod - 1) * np.pi - eps]
        arr = arr_temp0
    else:  # contract the non-theta array using the theta array brr
        eps = 1e-11
        arr_temp0 = arr[brr <= (2 * nperiod - 1) * np.pi + eps]
        brr_temp0 = brr[brr <= (2 * nperiod - 1) * np.pi + eps]
        arr_temp0 = arr_temp0[brr_temp0 >= -(2 * nperiod - 1) * np.pi - eps]
        brr_temp0 = brr_temp0[brr_temp0 >= -(2 * nperiod - 1) * np.pi - eps]
        arr = arr_temp0

    return arr


def dermv(arr, brr, par="e"):
    """
    Finite difference subroutine for a non-uniform grid.
    Calculates d(arr)/d(brr) using a second-order finite difference method.
    brr can be a non-uniformly spaced array
    Inputs
    ------
    arr: numpy array
    Input array of the dependent variable
    brr: numpy array
    Input array of the independent variable
    par: str
    Expected parity of the output array, par = 'e' means even par = 'o' means odd

    Returns
    -------
    The derivative d(arr)/d(brr)
    """
    temp = np.shape(arr)
    if len(temp) == 1:  # One dimensional input array
        if par == "e":
            d1, d2 = np.shape(arr)[0], 1
            arr = np.reshape(arr, (d2, d1))
            brr = np.reshape(brr, (d2, d1))
            diff_arr = np.zeros((d2, d1))
            diff_arr[0, 0] = 0.0  # (arr_theta_-0 - arr_theta_+0)  = 0
            diff_arr[0, -1] = 0.0
            # diff_arr[0, 1:-1] = (
            #     np.diff(arr[0,:-1], axis=0) + np.diff(arr[0,1:], axis=0)
            # )
            for i in range(1, d1 - 1):
                h1 = brr[0, i + 1] - brr[0, i]
                h0 = brr[0, i] - brr[0, i - 1]
                diff_arr[0, i] = (
                    arr[0, i + 1] / h1**2
                    + arr[0, i] * (1 / h0**2 - 1 / h1**2)
                    - arr[0, i - 1] / h0**2
                ) / (1 / h1 + 1 / h0)
        else:
            d1, d2 = np.shape(arr)[0], 1
            arr = np.reshape(arr, (d2, d1))
            brr = np.reshape(brr, (d2, d1))
            diff_arr = np.zeros((d2, d1))

            h1 = np.abs(brr[0, 1]) - np.abs(brr[0, 0])
            h0 = np.abs(brr[0, -1]) - np.abs(brr[0, -2])
            diff_arr[0, 0] = (4 * arr[0, 1] - 3 * arr[0, 0] - arr[0, 2]) / (
                2 * (brr[0, 1] - brr[0, 0])
            )

            diff_arr[0, -1] = (-4 * arr[0, -2] + 3 * arr[0, -1] + arr[0, -3]) / (
                2 * (brr[0, -1] - brr[0, -2])
            )
            for i in range(1, d1 - 1):
                h1 = brr[0, i + 1] - brr[0, i]
                h0 = brr[0, i] - brr[0, i - 1]
                diff_arr[0, i] = (
                    arr[0, i + 1] / h1**2
                    + arr[0, i] * (1 / h0**2 - 1 / h1**2)
                    - arr[0, i - 1] / h0**2
                ) / (1 / h1 + 1 / h0)
        diff_arr = diff_arr[0]

    else:
        d1, d2, d3 = np.shape(arr)
        diff_arr = np.zeros((d1, d2, d3))
        if par == "e":  # Even parity
            diff_arr[:, :, 0] = np.zeros((d1, d2))
            diff_arr[:, :, -1] = np.zeros((d1, d2))
            for i in range(1, d3 - 1):
                h1 = brr[:, :, i + 1] - brr[:, :, i]
                h0 = brr[:, :, i] - brr[:, :, i - 1]
                diff_arr[:, :, i] = (
                    arr[:, :, i + 1] / h1**2
                    + arr[:, :, i] * (1 / h0**2 - 1 / h1**2)
                    - arr[:, :, i - 1] / h0**2
                ) / (1 / h1 + 1 / h0)
        else:
            diff_arr[:, :, 0] = (
                2 * (arr[:, :, 1] - arr[:, :, 0]) / (2 * (brr[:, :, 1] - brr[:, :, 0]))
            )
            diff_arr[:, :, -1] = (
                2
                * (arr[:, :, -1] - arr[:, :, -2])
                / (2 * (brr[:, :, -1] - brr[:, :, -2]))
            )
            for i in range(1, d3 - 1):
                h1 = brr[:, :, i + 1] - brr[:, :, i]
                h0 = brr[:, :, i] - brr[:, :, i - 1]
                diff_arr[:, :, i] = (
                    arr[:, :, i + 1] / h1**2
                    + arr[:, :, i] * (1 / h0**2 - 1 / h1**2)
                    - arr[:, :, i - 1] / h0**2
                ) / (1 / h1 + 1 / h0)

    return diff_arr


####################################################################################################################
###########################-------------------------3D EQUILIBRIUM----------------------############################
####################################################################################################################


class Struct:
    """
    This class is just a dummy mutable object to which we can add attributes.
    """


def vmec_splines(nc_obj, booz_obj):
    """
    Initialize radial splines for a VMEC equilibrium.

    Args:
        vmec: a netCDF object

    Returns:
        A structure with the splines as attributes.
    """
    results = Struct()

    rmnc_b = []
    zmns_b = []
    numns_b = []

    d_rmnc_b_d_s = []
    d_zmns_b_d_s = []
    d_numns_b_d_s = []

    ns = nc_obj.variables["ns"][:].data
    s_full_grid = np.linspace(0, 1, ns)
    s_half_grid = s_full_grid[1:] - 0.5 * np.diff(s_full_grid)[0]

    # Boozer quantities are calculated on the half grid by booz_xform
    for jmn in range(int(booz_obj.mnboz)):
        rmnc_b.append(
            InterpolatedUnivariateSpline(s_half_grid, booz_obj.rmnc_b.T[:, jmn])
        )
        zmns_b.append(
            InterpolatedUnivariateSpline(s_half_grid, booz_obj.zmns_b.T[:, jmn])
        )
        numns_b.append(
            InterpolatedUnivariateSpline(s_half_grid, booz_obj.numns_b.T[:, jmn])
        )

        d_rmnc_b_d_s.append(rmnc_b[-1].derivative())
        d_zmns_b_d_s.append(zmns_b[-1].derivative())
        d_numns_b_d_s.append(numns_b[-1].derivative())

    gmnc_b = []
    bmnc_b = []
    d_bmnc_b_d_s = []

    for jmn in range(int(booz_obj.mnboz)):
        gmnc_b.append(
            InterpolatedUnivariateSpline(s_half_grid, booz_obj.gmnc_b.T[:, jmn])
        )
        bmnc_b.append(
            InterpolatedUnivariateSpline(s_half_grid, booz_obj.bmnc_b.T[:, jmn])
        )
        d_bmnc_b_d_s.append(bmnc_b[-1].derivative())

    results.Gfun = InterpolatedUnivariateSpline(s_half_grid, booz_obj.Boozer_G)
    results.Ifun = InterpolatedUnivariateSpline(s_half_grid, booz_obj.Boozer_I)

    # Useful 1d profiles:
    results.pressure = InterpolatedUnivariateSpline(
        s_half_grid, nc_obj.variables["pres"][1:]
    )
    results.d_pressure_d_s = results.pressure.derivative()
    results.psi = InterpolatedUnivariateSpline(
        s_half_grid, nc_obj.variables["phi"][1:] / (2 * np.pi)
    )
    results.d_psi_d_s = results.psi.derivative()
    results.iota = InterpolatedUnivariateSpline(
        s_half_grid, nc_obj.variables["iotas"][1:]
    )
    results.d_iota_d_s = results.iota.derivative()

    # Save other useful quantities:
    results.phiedge = nc_obj.variables["phi"][-1].data
    variables = ["Aminor_p", "nfp", "raxis_cc", "mpol", "ntor"]
    for v in variables:
        results.__setattr__(v, eval("nc_obj.variables['" + v + "'][:].data"))

    variables1 = ["xm_b", "xn_b", "xm_nyq_b", "xn_nyq_b", "mnbooz", "mboz", "nboz"]
    variables2 = ["xm_b", "xn_b", "xm_b", "xn_b", "mnboz", "mboz", "nboz"]
    for k, v in enumerate(variables1):
        results.__setattr__(v, eval("booz_obj." + variables2[k]))
        # print(v, k)

    # pdb.set_trace()
    variables = [
        "rmnc_b",
        "zmns_b",
        "numns_b",
        "d_rmnc_b_d_s",
        "d_zmns_b_d_s",
        "d_numns_b_d_s",
        "gmnc_b",
        "bmnc_b",
        "d_bmnc_b_d_s",
    ]
    for v in variables:
        results.__setattr__(v, eval(v))

    return results


#########################################################################################################
#######################------------------GEOMETRY CALCULATION FUN--------------------####################
#########################################################################################################


def vmec_fieldlines_axisym(
    vmec_fname,
    s,
    alpha,
    theta1d=None,
    phi1d=None,
    phi_center=0,
    sfac=1,
    pfac=1,
    isaxisym=0,
):
    """
    Geometry routine for GX/GS2.
    Takes in a 1D theta or phi array in boozer coordinates, an array of flux surfaces,
    another array of field line labels and generates the coefficients needed for a local
    stability analysis.
    Additionally, this routine allows the user to self-consistently change the local pressure gradient and average shear and recalculates the geometric coefficients.
    Inputs:
    ------
    s: List or numpy array
    The normalized toroidal flux psi/psi_boundary
    alpha: list or numpy array
    alpha = theta_b - iota * phi_b is the field line label
    theta1d: numpy array
    Boozer theta
    phi1d: numpy array
    Boozer phi
    sfac: float
    Local variation of the average shear, Geometry is calculated for a total shear of shear*sfac.
    So if want to calculate geometry at 2.5 x nominal shear, sfac = 1.5.
    sfac: float
    Local variation of the pressure gradient. Geometry is calculated for a pressure gradient of dpds*pfac

    Outputs
    -------
    gds22: numpy array
    Flux expansion term
    gds21 numpy array
    Integrated local shear
    gds2: numpy array
    Field line bending
    bmag: numpy array
    normalized magnetic field strength
    gradpar: numpy array
    Parallel gradient b dot grad phi
    gbdrift: numpy array
    Grad-B drift geometry factor
    cvdrift: numpy array
    Curvature drift geometry factor
    cvdrift0: numpy array
    theta_PEST: numpy array
    theta_PEST for the given theta_b array.
    theta_geo: numpy array
    geometric (arctan) theta for the given theta_b array.
    """
    nc_obj = ds(vmec_fname, "r")

    mpol = nc_obj.variables["mpol"][:].data
    ntor = nc_obj.variables["ntor"][:].data

    booz_obj = bxform.Booz_xform()
    booz_obj.read_wout(vmec_fname)
    booz_obj.mboz = int(2 * mpol)
    booz_obj.nboz = int(2 * ntor)
    booz_obj.run()

    vs = vmec_splines(nc_obj, booz_obj)

    # Make sure s is an array:
    try:
        ns = len(s)
    except:
        s = [s]
    s = np.array(s)
    ns = len(s)

    # Make sure alpha is an array
    # For axisymmetric equilibria, all field lines are identical, i.e., your choice of alpha doesn't matter
    try:
        nalpha = len(alpha)
    except:
        alpha = [alpha]
    alpha = np.array(alpha)
    nalpha = len(alpha)

    if (theta1d is not None) and (phi1d is not None):
        raise ValueError("You cannot specify both theta and phi")
    if (theta1d is None) and (phi1d is None):
        raise ValueError("You must specify either theta or phi")
    if theta1d is None:
        nl = len(phi1d)
    else:
        nl = len(theta1d)

    # Now that we have an s grid, evaluate everything on that grid:
    d_pressure_d_s = vs.d_pressure_d_s(s)
    d_psi_d_s = vs.d_psi_d_s(s)
    iota = vs.iota(s)
    d_iota_d_s = vs.d_iota_d_s(s)
    shat = (-2 * s / iota) * d_iota_d_s  # depends on the definitn of rho

    L_reference = vs.Aminor_p

    edge_toroidal_flux_over_2pi = -vs.phiedge / (2 * np.pi)
    toroidal_flux_sign = np.sign(edge_toroidal_flux_over_2pi)
    B_reference = 2 * abs(edge_toroidal_flux_over_2pi) / (L_reference * L_reference)

    xm_b = vs.xm_b
    xn_b = vs.xn_b
    mnmax_b = vs.mnbooz

    G = vs.Gfun(s)
    d_G_d_s = vs.Gfun.derivative()(s)
    I = vs.Ifun(s)
    d_I_d_s = vs.Ifun.derivative()(s)

    rmnc_b = np.zeros((ns, mnmax_b))
    zmns_b = np.zeros((ns, mnmax_b))
    numns_b = np.zeros((ns, mnmax_b))
    d_rmnc_b_d_s = np.zeros((ns, mnmax_b))
    d_zmns_b_d_s = np.zeros((ns, mnmax_b))
    d_numns_b_d_s = np.zeros((ns, mnmax_b))

    numns_b = np.zeros((ns, mnmax_b))
    gmnc_b = np.zeros((ns, mnmax_b))
    bmnc_b = np.zeros((ns, mnmax_b))
    d_bmnc_b_d_s = np.zeros((ns, mnmax_b))

    delmnc_b = np.zeros((ns, mnmax_b))
    lambmnc_b = np.zeros((ns, mnmax_b))
    betamns_b = np.zeros((ns, mnmax_b))

    theta_b = np.zeros((ns, nalpha, nl))
    phi_b = np.zeros((ns, nalpha, nl))

    Vprime = np.zeros((ns, 1))

    for jmn in range(mnmax_b):
        rmnc_b[:, jmn] = vs.rmnc_b[jmn](s)
        zmns_b[:, jmn] = vs.zmns_b[jmn](s)
        numns_b[:, jmn] = vs.numns_b[jmn](s)
        d_rmnc_b_d_s[:, jmn] = vs.d_rmnc_b_d_s[jmn](s)
        d_zmns_b_d_s[:, jmn] = vs.d_zmns_b_d_s[jmn](s)
        d_numns_b_d_s[:, jmn] = vs.d_numns_b_d_s[jmn](s)
        gmnc_b[:, jmn] = vs.gmnc_b[jmn](s)
        bmnc_b[:, jmn] = vs.bmnc_b[jmn](s)
        d_bmnc_b_d_s[:, jmn] = vs.d_bmnc_b_d_s[jmn](s)

    if theta1d is None:
        # We are given phi_boozer. Compute theta_boozer
        for js in range(ns):
            phi_b[js, :, :] = phi1d[None, :]
            theta_b[js, :, :] = alpha[:, None] + iota[js] * (phi1d[None, :])
    else:
        # We are given theta_pest. Compute phi:
        for js in range(ns):
            theta_b[js, :, :] = theta1d[None, :]
            phi_b[js, :, :] = (theta1d[None, :] - alpha[:, None]) / iota[js]

    # Now that we know theta_boozer, compute all the geometric quantities
    angle_b = (
        xm_b[:, None, None, None] * (theta_b[None, :, :, :])
        - xn_b[:, None, None, None] * phi_b[None, :, :, :]
    )
    cosangle_b = np.cos(angle_b)
    sinangle_b = np.sin(angle_b)

    R_b = np.einsum("ij,jikl->ikl", rmnc_b, cosangle_b)
    Z_b = np.einsum("ij,jikl->ikl", zmns_b, sinangle_b)

    flipit = 0.0

    # if R is increasing AND Z is decreasing, we must be moving counter clockwise from
    # the inboard side, otherwise we need to flip the theta coordinate
    if R_b[0][0][0] > R_b[0][0][1] or Z_b[0][0][1] > Z_b[0][0][0]:
        flipit = 1

    # if an equilibrium is 3D, we disable flipit
    if isaxisym == 0:
        flipit == 0

    R_mag_ax = vs.raxis_cc[0]

    #####################################################################################
    #####################------------BOOZER CALCULATIONS--------------###################
    #####################################################################################

    if flipit == 1:
        angle_b = (
            xm_b[:, None, None, None] * (theta_b[None, :, :, :] + np.pi)
            - xn_b[:, None, None, None] * phi_b
        )
    else:
        angle_b = (
            xm_b[:, None, None, None] * theta_b[None, :, :, :]
            - xn_b[:, None, None, None] * phi_b
        )

    cosangle_b = np.cos(angle_b)
    sinangle_b = np.sin(angle_b)
    mcosangle_b = xm_b[:, None, None, None] * cosangle_b
    ncosangle_b = xn_b[:, None, None, None] * cosangle_b
    msinangle_b = xm_b[:, None, None, None] * sinangle_b
    nsinangle_b = xn_b[:, None, None, None] * sinangle_b
    # Order of indices in cosangle_b and sinangle_b: mn_b, s, alpha, l
    # Order of indices in rmnc, bmnc, etc: s, mn_b
    R_b = np.einsum("ij,jikl->ikl", rmnc_b, cosangle_b)
    d_R_b_d_s = np.einsum("ij,jikl->ikl", d_rmnc_b_d_s, cosangle_b)
    d_R_b_d_theta_b = -np.einsum("ij,jikl->ikl", rmnc_b, msinangle_b)
    d_R_b_d_phi_b = np.einsum("ij,jikl->ikl", rmnc_b, nsinangle_b)

    Z_b = np.einsum("ij,jikl->ikl", zmns_b, sinangle_b)
    d_Z_b_d_s = np.einsum("ij,jikl->ikl", d_zmns_b_d_s, sinangle_b)
    d_Z_b_d_theta_b = np.einsum("ij,jikl->ikl", zmns_b, mcosangle_b)
    d_Z_b_d_phi_b = -np.einsum("ij,jikl->ikl", zmns_b, ncosangle_b)

    nu_b = np.einsum("ij,jikl->ikl", numns_b, sinangle_b)
    d_nu_b_d_s = np.einsum("ij,jikl->ikl", d_numns_b_d_s, sinangle_b)
    d_nu_b_d_theta_b = np.einsum("ij,jikl->ikl", numns_b, mcosangle_b)
    d_nu_b_d_phi_b = -np.einsum("ij,jikl->ikl", numns_b, ncosangle_b)

    # sqrt_g_booz = (G + iota * I)/B**2
    sqrt_g_booz = np.einsum("ij,jikl->ikl", gmnc_b, cosangle_b)
    d_sqrt_g_booz_d_theta_b = -np.einsum("ij,jikl->ikl", gmnc_b, msinangle_b)
    d_sqrt_g_booz_d_phi_b = np.einsum("ij,jikl->ikl", gmnc_b, nsinangle_b)
    modB_b = np.einsum("ij,jikl->ikl", bmnc_b, cosangle_b)
    d_B_b_d_s = np.einsum("ij,jikl->ikl", d_bmnc_b_d_s, cosangle_b)
    d_B_b_d_theta_b = -np.einsum("ij,jikl->ikl", bmnc_b, msinangle_b)
    d_B_b_d_phi_b = np.einsum("ij,jikl->ikl", bmnc_b, nsinangle_b)

    Vprime = gmnc_b[:, 0]

    delmnc_b[:, 1:] = gmnc_b[:, 1:] / Vprime[:, None]
    betamns_b[:, 1:] = (
        delmnc_b[:, 1:]
        * 1
        / edge_toroidal_flux_over_2pi
        * mu_0
        * d_pressure_d_s[:, None]
        * Vprime[:, None]
        / (xm_b[1:] * iota[:, None] - xn_b[1:])
    )
    lambmnc_b[:, 1:] = (
        delmnc_b[:, 1:]
        * (xm_b[1:] * G[:, None] + xn_b[1:] * I[:, None])
        / (
            (xm_b[1:] * iota[:, None] - xn_b[1:])
            * (G[:, None] + iota[:, None] * I[:, None])
        )
    )

    beta_b = np.einsum("ij,jikl->ikl", betamns_b, sinangle_b)
    lambda_b = np.einsum("ij,jikl->ikl", lambmnc_b, cosangle_b)

    ###############################################################################
    # Using R(theta,phi) and Z(theta,phi), compute the Cartesian
    # components of the gradient basis vectors using the dual relations:
    # This calculation is done in Boozer coordinates
    ###############################################################################
    phi_cyl = phi_b - nu_b
    sinphi = np.sin(phi_cyl)
    cosphi = np.cos(phi_cyl)
    # X = R * cos(phi):
    d_X_d_theta_b = d_R_b_d_theta_b * cosphi - R_b * sinphi * (-1 * d_nu_b_d_theta_b)
    d_X_d_phi_b = d_R_b_d_phi_b * cosphi - R_b * sinphi * (1 - d_nu_b_d_phi_b)
    d_X_d_s = d_R_b_d_s * cosphi - R_b * sinphi * (-1 * d_nu_b_d_s)
    # Y = R * sin(phi):
    d_Y_d_theta_b = d_R_b_d_theta_b * sinphi + R_b * cosphi * (-1 * d_nu_b_d_theta_b)
    d_Y_d_phi_b = d_R_b_d_phi_b * sinphi + R_b * cosphi * (1 - d_nu_b_d_phi_b)
    d_Y_d_s = d_R_b_d_s * sinphi + R_b * cosphi * (-1 * d_nu_b_d_s)

    # Dual relations
    grad_psi_X = (
        d_Y_d_theta_b * d_Z_b_d_phi_b - d_Z_b_d_theta_b * d_Y_d_phi_b
    ) / sqrt_g_booz
    grad_psi_Y = (
        d_Z_b_d_theta_b * d_X_d_phi_b - d_X_d_theta_b * d_Z_b_d_phi_b
    ) / sqrt_g_booz
    grad_psi_Z = (
        d_X_d_theta_b * d_Y_d_phi_b - d_Y_d_theta_b * d_X_d_phi_b
    ) / sqrt_g_booz

    g_sup_psi_psi = grad_psi_X**2 + grad_psi_Y**2 + grad_psi_Z**2

    # Check varible names
    grad_theta_b_X = (
        (d_Y_d_phi_b * d_Z_b_d_s - d_Z_b_d_phi_b * d_Y_d_s)
        / sqrt_g_booz
        * 1
        / edge_toroidal_flux_over_2pi
    )
    grad_theta_b_Y = (
        (d_Z_b_d_phi_b * d_X_d_s - d_X_d_phi_b * d_Z_b_d_s)
        / sqrt_g_booz
        * 1
        / edge_toroidal_flux_over_2pi
    )
    grad_theta_b_Z = (
        (d_X_d_phi_b * d_Y_d_s - d_Y_d_phi_b * d_X_d_s)
        / sqrt_g_booz
        * 1
        / edge_toroidal_flux_over_2pi
    )

    grad_phi_b_X = (
        (d_Y_d_s * d_Z_b_d_theta_b - d_Z_b_d_s * d_Y_d_theta_b)
        / sqrt_g_booz
        * 1
        / edge_toroidal_flux_over_2pi
    )
    grad_phi_b_Y = (
        (d_Z_b_d_s * d_X_d_theta_b - d_X_d_s * d_Z_b_d_theta_b)
        / sqrt_g_booz
        * 1
        / edge_toroidal_flux_over_2pi
    )
    grad_phi_b_Z = (
        (d_X_d_s * d_Y_d_theta_b - d_Y_d_s * d_X_d_theta_b)
        / sqrt_g_booz
        * 1
        / edge_toroidal_flux_over_2pi
    )

    grad_alpha_X = (
        -phi_b * d_iota_d_s[:, None, None] * grad_psi_X / edge_toroidal_flux_over_2pi
        + grad_theta_b_X
        - iota[:, None, None] * grad_phi_b_X
    )
    grad_alpha_Y = (
        -phi_b * d_iota_d_s[:, None, None] * grad_psi_Y / edge_toroidal_flux_over_2pi
        + grad_theta_b_Y
        - iota[:, None, None] * grad_phi_b_Y
    )
    grad_alpha_Z = (
        -phi_b * d_iota_d_s[:, None, None] * grad_psi_Z / edge_toroidal_flux_over_2pi
        + grad_theta_b_Z
        - iota[:, None, None] * grad_phi_b_Z
    )

    #######################################################################################
    ##############------------LOCAL VARIATION OF A 3D EQUILIBRIUM------------##############
    #######################################################################################
    # Calculating the coefficients D1 and D2 needed for Hegna-Nakajima calculation
    # NOTE: 2D functions do not require the alpha dimension. Remove it later.
    # NOTE: This calculation needs to be wrapped in a loop over ns (flux surfaces)
    ## Full flux surface average of various quantities needed to calculate D_HNGC
    ntheta_grid = 401
    nphi_grid = 401
    theta_b_grid = np.linspace(-np.pi, np.pi, ntheta_grid)
    phi_b_grid = np.linspace(-np.pi, np.pi, nphi_grid)
    th_b_2D, ph_b_2D = np.meshgrid(theta_b_grid, phi_b_grid)

    if flipit == 1:
        angle_b_2D = (
            xm_b[:, None, None, None, None] * (th_b_2D[None, None, None, :, :] + np.pi)
            - xn_b[:, None, None, None, None] * ph_b_2D[None, None, None, :, :]
        )
    else:
        angle_b_2D = (
            xm_b[:, None, None, None, None] * (th_b_2D[None, None, None, :, :])
            - xn_b[:, None, None, None, None] * ph_b_2D[None, None, None, :, :]
        )

    cosangle_b_2D = np.cos(angle_b_2D)
    sinangle_b_2D = np.sin(angle_b_2D)

    mcosangle_b_2D = xm_b[:, None, None, None, None] * cosangle_b_2D
    ncosangle_b_2D = xn_b[:, None, None, None, None] * cosangle_b_2D
    msinangle_b_2D = xm_b[:, None, None, None, None] * sinangle_b_2D
    nsinangle_b_2D = xn_b[:, None, None, None, None] * sinangle_b_2D

    lambda_b_2D = np.einsum("ij,jiklm->iklm", lambmnc_b, cosangle_b_2D)

    R_b_2D = np.einsum("ij,jiklm->iklm", rmnc_b, cosangle_b_2D)
    d_R_b_d_s_2D = np.einsum("ij,jiklm->iklm", d_rmnc_b_d_s, cosangle_b_2D)
    d_R_b_d_theta_b_2D = -np.einsum("ij,jiklm->iklm", rmnc_b, msinangle_b_2D)
    d_R_b_d_phi_b_2D = np.einsum("ij,jiklm->iklm", rmnc_b, nsinangle_b_2D)

    Z_b_2D = np.einsum("ij,jiklm->iklm", zmns_b, sinangle_b_2D)
    d_Z_b_d_s_2D = np.einsum("ij,jiklm->iklm", d_zmns_b_d_s, sinangle_b_2D)
    d_Z_b_d_theta_b_2D = np.einsum("ij,jiklm->iklm", zmns_b, mcosangle_b_2D)
    d_Z_b_d_phi_b_2D = -np.einsum("ij,jiklm->iklm", zmns_b, ncosangle_b_2D)

    nu_b_2D = np.einsum("ij,jiklm->iklm", numns_b, sinangle_b_2D)
    d_nu_b_d_s_2D = np.einsum("ij,jiklm->iklm", d_numns_b_d_s, sinangle_b_2D)
    d_nu_b_d_theta_b_2D = np.einsum("ij,jiklm->iklm", numns_b, mcosangle_b_2D)
    d_nu_b_d_phi_b_2D = -np.einsum("ij,jiklm->iklm", numns_b, ncosangle_b_2D)

    sqrt_g_booz_2D = np.einsum("ij,jiklm->iklm", gmnc_b, cosangle_b_2D)
    d_sqrt_g_booz_d_theta_b_2D = -np.einsum("ij,jiklm->iklm", gmnc_b, msinangle_b_2D)
    d_sqrt_g_booz_d_phi_b_2D = np.einsum("ij,jiklm->iklm", gmnc_b, nsinangle_b_2D)
    modB_b_2D = np.einsum("ij,jiklm->iklm", bmnc_b, cosangle_b_2D)

    # *********************************************************************
    # Using R(theta,phi) and Z(theta,phi), compute the Cartesian
    # components of the gradient basis vectors using the dual relations:
    # This calculation is done in Boozer coordinates
    # *********************************************************************
    ph_nat_2D = ph_b_2D - nu_b_2D
    sinphi_2D = np.sin(ph_nat_2D)
    cosphi_2D = np.cos(ph_nat_2D)
    # X = R * cos(phi):
    d_X_d_th_b_2D = d_R_b_d_theta_b_2D * cosphi_2D - R_b_2D * sinphi_2D * (
        -1 * d_nu_b_d_theta_b_2D
    )
    d_X_d_phi_2D = d_R_b_d_phi_b_2D * cosphi_2D - R_b_2D * sinphi_2D * (
        1 - d_nu_b_d_phi_b_2D
    )
    # Y = R * sin(phi):
    d_Y_d_th_b_2D = d_R_b_d_theta_b_2D * sinphi_2D + R_b_2D * cosphi_2D * (
        -1 * d_nu_b_d_theta_b_2D
    )
    d_Y_d_phi_2D = d_R_b_d_phi_b_2D * sinphi_2D + R_b_2D * cosphi_2D * (
        1 - d_nu_b_d_phi_b_2D
    )

    # Now use the dual relations to get the Cartesian components of grad s, grad theta_vmec, and grad phi:
    grad_psi_X_2D = (
        d_Y_d_th_b_2D * d_Z_b_d_phi_b_2D - d_Z_b_d_theta_b_2D * d_Y_d_phi_2D
    ) / sqrt_g_booz_2D
    grad_psi_Y_2D = (
        d_Z_b_d_theta_b_2D * d_X_d_phi_2D - d_X_d_th_b_2D * d_Z_b_d_phi_b_2D
    ) / sqrt_g_booz_2D
    grad_psi_Z_2D = (
        d_X_d_th_b_2D * d_Y_d_phi_2D - d_Y_d_th_b_2D * d_X_d_phi_2D
    ) / sqrt_g_booz_2D

    g_sup_psi_psi_2D = grad_psi_X_2D**2 + grad_psi_Y_2D**2 + grad_psi_Z_2D**2
    g_sup_psi_psi_2D_inv = 1 / g_sup_psi_psi_2D

    lam_over_g_sup_psi_psi_2D = lambda_b_2D * g_sup_psi_psi_2D_inv

    # Flux surface integrals D1 and D2 are needed to locally vary the gradients of a 3D equilibrium.
    D1 = (
        simps(
            [
                simps(g_sup_psi_psi_1D_inv, theta_b_grid)
                for g_sup_psi_psi_1D_inv in g_sup_psi_psi_2D_inv[0][0]
            ],
            phi_b_grid,
        )
        / (2 * np.pi) ** 2
    )

    D2 = (
        simps(
            [
                simps(lam_over_g_sup_psi_psi_1D, theta_b_grid)
                for lam_over_g_sup_psi_psi_1D in lam_over_g_sup_psi_psi_2D[0][0]
            ],
            phi_b_grid,
        )
        / (2 * np.pi) ** 2
    )

    ## EQUILIBRIUM CHECK: Flux surface averaged MHD force balance equation.
    np.testing.assert_allclose(
        d_G_d_s[:, None, None]
        + iota[:, None, None] * d_I_d_s[:, None, None]
        + mu_0 * d_pressure_d_s[:, None, None] * Vprime,
        0,
        rtol=1e-5,
        atol=1e-5,
    )

    theta_b_trunc = nperiod_set(theta_b, 1)
    phi_b_trunc = nperiod_set(phi_b, 1, brr=theta_b)
    g_sup_psi_psi_trunc = nperiod_set(g_sup_psi_psi, 1, brr=theta_b)
    lambda_b_trunc = nperiod_set(lambda_b, 1, brr=theta_b)

    # integrated inverse flux expansion term
    intinv_g_sup_psi_psi = ctrap(1 / g_sup_psi_psi, phi_b, initial=0)
    int_lambda_div_g_sup_psi_psi = ctrap(lambda_b / g_sup_psi_psi, phi_b, initial=0)

    print("theta_0 needed here!")
    theta_0 = 0
    phi_0 = theta_0 / abs(iota)
    print("Make a function of ns and nalpha")
    spl0 = InterpolatedUnivariateSpline(theta_b[0][0], intinv_g_sup_psi_psi[0][0])
    intinv_g_sup_psi_psi = intinv_g_sup_psi_psi - spl0(theta_0)

    spl1 = InterpolatedUnivariateSpline(
        theta_b[0][0], int_lambda_div_g_sup_psi_psi[0][0]
    )
    int_lambda_div_g_sup_psi_psi = int_lambda_div_g_sup_psi_psi - spl1(theta_0)

    # Additional shear and pressure gradient (in addn. to nominal)
    d_iota_d_s_1 = -(iota / (2 * s)) * (sfac - 1) * shat * np.ones((ns,))
    d_pressure_d_s_1 = mu_0 * (pfac - 1) * d_pressure_d_s * np.ones((ns,))
    # The deformation term from Hegna-Nakajima and Green-Chance papers
    D_HNGC = (
        1
        / edge_toroidal_flux_over_2pi
        * (
            d_iota_d_s_1[:, None, None] * (intinv_g_sup_psi_psi / D1 - (phi_b - phi_0))
            - d_pressure_d_s_1[:, None, None]
            * Vprime[:, None, None]
            * (G[:, None, None] + iota[:, None, None] * I[:, None, None])
            * (int_lambda_div_g_sup_psi_psi - D2 * intinv_g_sup_psi_psi / D1)
        )
    )

    # Now we recalculate some of the geometric coefficients in Boozer coordinates
    # Partially calculated in boozer coordinates
    grad_alpha_dot_grad_psi = (
        grad_alpha_X * grad_psi_X
        + grad_alpha_Y * grad_psi_Y
        + grad_alpha_Z * grad_psi_Z
    )

    # Intergrated local shear L1 is calculated using covariant basis
    # expressions in Hegna and Nakajima
    # We remove the secular part form the integrated local shear grad_alpha_dot_grad_psi_alt
    # NOTE: Potential sign issue here
    L0 = -1 * (
        grad_alpha_dot_grad_psi / g_sup_psi_psi
        + 1 / edge_toroidal_flux_over_2pi * d_iota_d_s[:, None, None] * phi_b
    )
    # L1 is the integrated local shear
    L1 = (
        -1 / edge_toroidal_flux_over_2pi * d_iota_d_s_1[:, None, None] * phi_b
        + grad_alpha_dot_grad_psi / g_sup_psi_psi
        - D_HNGC
    )

    L2 = d_iota_d_s[:, None, None] * 1 / edge_toroidal_flux_over_2pi

    # Normal curvature
    # NOTE: Test a case close to a rational surface to check the sign of beta_b
    kappa_n = (
        1
        / modB_b**2
        * (modB_b * d_B_b_d_s + mu_0 * d_pressure_d_s[:, None, None])
        * 1
        / edge_toroidal_flux_over_2pi
        - beta_b
        / (
            2
            * sqrt_g_booz
            * (G[:, None, None] + iota[:, None, None] * I[:, None, None])
        )
        * d_sqrt_g_booz_d_phi_b
        + L0
        * (G[:, None] * d_sqrt_g_booz_d_theta_b - I[:, None] * d_sqrt_g_booz_d_phi_b)
        / (2 * sqrt_g_booz * (G[:, None] + iota[:, None] * I[:, None]))
    )

    # Geodesic curvature
    kappa_g = (
        G[:, None] * d_sqrt_g_booz_d_theta_b - I[:, None] * d_sqrt_g_booz_d_phi_b
    ) / (2 * sqrt_g_booz * (G[:, None] + iota[:, None] * I[:, None]))

    B_cross_kappa_dot_grad_alpha_b = (kappa_n + kappa_g * L1) * modB_b**2

    B_cross_kappa_dot_grad_psi_b = kappa_g * modB_b**2

    grad_alpha_dot_grad_alpha_b = modB_b**2 / g_sup_psi_psi + g_sup_psi_psi * L1**2
    grad_alpha_dot_grad_psi_b = g_sup_psi_psi * L1
    grad_psi_dot_grad_psi_b = g_sup_psi_psi * L2

    sqrt_s = np.sqrt(s)

    L_reference = vs.Aminor_p
    toroidal_flux_sign = np.sign(edge_toroidal_flux_over_2pi)
    sqrt_s = np.sqrt(s)

    gbdrift0 = (
        -1.0
        * B_cross_kappa_dot_grad_psi_b
        * 2
        * shat[:, None, None]
        / (modB_b * modB_b * modB_b * sqrt_s[:, None, None])
        * toroidal_flux_sign
    )
    cvdrift0 = gbdrift0

    # Rprime_b = (
    #    np.interp(theta_b, d_R_b_d_s)
    #    * 1
    #    / edge_toroidal_flux_over_2pi
    #    * 1
    #    / iota
    #    * R_1
    #    * B_p_1
    # )
    # Zprime_1 = (
    #    np.interp(theta1d, theta_pest[0][0], d_Z_d_s[0][0])
    #    * 1
    #    / edge_toroidal_flux_over_2pi
    #    * 1
    #    / iota
    #    * R_1
    #    * B_p_1
    # )

    ## Now we calculate the same set of quantities in boozer coordinates after varying the
    ## local gradients.

    bmag = modB_b / B_reference
    # Is should be possible to avoid B_sup quantities altogether.
    gradpar_theta = (
        -1 / (B_reference * L_reference) * 1 / sqrt_g_booz * 1 / iota[:, None, None]
    )
    gradpar_phi = 1 / (B_reference * L_reference) * 1 / sqrt_g_booz

    gds2 = grad_alpha_dot_grad_alpha_b * L_reference * L_reference * s[:, None, None]
    gds21 = grad_alpha_dot_grad_psi_b * shat[:, None, None] / B_reference
    gds22 = (
        g_sup_psi_psi
        * shat[:, None, None]
        * shat[:, None, None]
        / (L_reference * L_reference * B_reference * B_reference * s[:, None, None])
    )

    gbdrift0 = (
        -1.0
        * B_cross_kappa_dot_grad_psi_b
        * 2
        * shat[:, None, None]
        / (modB_b * modB_b * modB_b * sqrt_s[:, None, None])
        * toroidal_flux_sign
    )

    cvdrift = (
        -1.0
        * 2
        * B_reference
        * L_reference
        * L_reference
        * sqrt_s[:, None, None]
        * B_cross_kappa_dot_grad_alpha_b
        / (modB_b * modB_b)
        * toroidal_flux_sign
    )

    gbdrift = cvdrift + 2 * B_reference * L_reference * L_reference * sqrt_s[
        :, None, None
    ] * mu_0 * d_pressure_d_s[:, None, None] * toroidal_flux_sign / (
        edge_toroidal_flux_over_2pi * modB_b * modB_b
    )

    cvdrift0 = gbdrift0

    # PEST theta; useful for comparison
    theta_PEST = theta_b - iota * nu_b

    # geometric theta; denotes the actual poloidal angle
    theta_geo = np.arctan2(Z_b, R_b - R_mag_ax)

    # This is half of the total beta_N. Used in GS2 as beta_ref
    beta_N = 4 * np.pi * 1e-7 * vs.pressure(s) / B_reference**2

    int_loc_shr = L0 + L1 + L2
    # Package results into a structure to return:
    results = Struct()
    variables = [
        "iota",
        "d_iota_d_s",
        "d_pressure_d_s",
        "d_psi_d_s",
        "shat",
        "alpha",
        "theta_b",
        "phi_b",
        "theta_PEST",
        "theta_geo",
        "edge_toroidal_flux_over_2pi",
        "R_b",
        "Rprime_b",
        "Z_b",
        "Zprime_b",
        "beta_N",
        "bmag",
        "gradpar_phi_b",
        "gds2_b",
        "gds21_b",
        "gds22_b",
        "gbdrift_b",
        "gbdrift0_b",
        "cvdrift_b",
        "cvdrift0_b",
        "grho_b",
    ]

    for v in variables:
        results.__setattr__(v, eval(v))

    return results




############################################################################################
###############--------------------CALCULATING GEOMETRY-------------------##################
############################################################################################

nt = 200
nperiod = 2
ntheta = 2 * nt * (2 * nperiod - 1) + 1
theta = np.linspace(-(2 * nperiod - 1) * np.pi, (2 * nperiod - 1) * np.pi, ntheta)
kxfac = abs(1.0)
rhoc = np.array([0.5])
alpha = 0.0

sfac = 2.0
pfac = 2.0

geo_coeffs = vmec_fieldlines(
    vmec_fname, rhoc, alpha, theta1d=theta, sfac=sfac, pfac=pfac, isaxisym=isaxisym
)

shat = geo_coeffs.shat
qfac = 1 / geo_coeffs.iota
bmag = geo_coeffs.bmag[0][0]
gradpar = abs(geo_coeffs.gradpar_theta[0][0])
cvdrift = geo_coeffs.cvdrift[0][0]
gbdrift = geo_coeffs.gbdrift[0][0]
gbdrift0 = geo_coeffs.gbdrift0[0][0]
cvdrift0 = geo_coeffs.cvdrift0[0][0]
gds2 = geo_coeffs.gds2[0][0]
gds21 = geo_coeffs.gds21[0][0]
gds22 = geo_coeffs.gds22[0][0]
R = geo_coeffs.R_b[0][0]
Z = geo_coeffs.Z_b[0][0]

# Must change this!
grho = gds22

dpsidrho = geo_coeffs.d_psi_d_s  # only true when rho = sqrt(psi)
drhodpsi = 1 / dpsidrho
Rmaj = (np.max(R) + np.min(R)) / 2


################################################################################################
##################------------------EQUAL-ARC THETA CALCULATION---------------##################
################################################################################################

theta_trun = nperiod_set(theta, 1, extend=False)
gradpar_trun = nperiod_set(gradpar, 1, extend=False, brr=theta)

gradpar_eqarc = 2 * np.pi / (ctrap(1 / gradpar_trun, theta_trun, initial=0)[-1])
theta_eqarc = ctrap(gradpar_eqarc / gradpar_trun, theta_trun, initial=0) - np.pi
theta_eqarc_extend = nperiod_set(theta_eqarc, nperiod, extend=True)
# theta_eqarc = np.linspace(-(2 * nperiod - 1) * np.pi, (2 * nperiod - 1) * np.pi, ntheta)
theta_eqarc = theta_eqarc_extend
theta_PEST = theta
## Check that the bounday conditions are always satisfied (theta = theta_eqarc at theta = +-pi)
bmag_eqarc = np.interp(theta_eqarc, theta_eqarc_extend, bmag)
gds21_eqarc = np.interp(theta_eqarc, theta_eqarc_extend, gds21)
cvdrift0_eqarc = np.interp(theta_eqarc, theta_eqarc_extend, cvdrift0)
gbdrift0_eqarc = np.interp(theta_eqarc, theta_eqarc_extend, gbdrift0)
R_eqarc = np.interp(theta_eqarc, theta_eqarc_extend, R)
Z_eqarc = np.interp(theta_eqarc, theta_eqarc_extend, Z)
gradpar_eqarc = gradpar_eqarc * np.ones((len(bmag_eqarc),))


################################################################################################
###################--------------------GX SAVE FORMAT1-------------------#######################
################################################################################################


A1 = []
A2 = []
A3 = []
A4 = []
A5 = []
A6 = []
A7 = []
A8 = []

for i in range(ntheta):
    # calculate grho
    A2.append(
        "    {0:.9e}    {1:.9e}    {2:.9e}     {3:.9e}\n".format(
            gbdrift_eqarc[i], gradpar_eqarc[i], gds22_eqarc[i], theta_eqarc[i]
        )
    )
    A3.append(
        "    {0:.9e}    {1:.9e}    {2:.12e}     {3:.9e}\n".format(
            cvdrift_eqarc[i], gds2_eqarc[i], bmag_eqarc[i], theta_eqarc[i]
        )
    )
    A4.append(
        "    {0:.9e}    {1:.9e}    {2:.9e}\n".format(
            gds21_eqarc[i], gds22_eqarc[i], theta_eqarc[i]
        )
    )
    A5.append(
        "    {0:.9e}    {1:.9e}    {2:.9e}\n".format(
            gbdrift0_eqarc[i], gbdrift0_eqarc[i], theta_eqarc[i]
        )
    )
    # MUST CHECK THESE FOR NONLINEAR RUNS!
    A6.append(
        "{0:.9e}    {1:.9e}    {2:.9e}\n".format(R_eqarc[i], R_eqarc[i], theta_eqarc[i])
    )
    A7.append(
        "{0:.9e}    {1:.9e}    {2:.9e}\n".format(Z_eqarc[i], Z_eqarc[i], theta_eqarc[i])
    )
    A8.append(
        "{0:.9e}    {1:.9e}    {2:.9e}\n".format(R_eqarc[i], R_eqarc[i], theta_eqarc[i])
    )


A1.append([A2, A3, A4, A5, A6, A7, A8])
A1 = A1[0]


print("Writing eikfile", eikfile)
g = open(eikfile, "w")

headings = [
    "ntgrid nperiod ntheta drhodpsi rmaj shat kxfac q\n",
    "gbdrift gradpar grho tgrid\n",
    "cvdrift gds2 bmag tgrid\n",
    "gds21 gds22 tgrid\n",
    "cvdrift0 gbdrift0 tgrid\n",
]

g.writelines(headings[0])


g.writelines(
    "  {0:d}    {1:d}    {2:d}   {3:.3f}   {4:.1f}    {5:.7f}   {6:.1f}   {7:.4f}\n".format(
        int((ntheta - 1) / 2),
        int(1),
        int(ntheta - 1),
        float(abs(1 / dpsidrho)),
        (np.max(R) + np.min(R)) / 2.0,
        shat.item(),
        (abs(qfac / rhoc * dpsidrho)).item(),
        qfac.item(),
    )
)

for i in np.arange(1, len(headings)):
    g.writelines(headings[i])
    for j in range(ntheta):
        g.write(A1[i - 1][j])
g.close()


################################################################################################
################---------------------GX SAVE FORMAT2---------------------------#################
################################################################################################
try:
    # import netCDF4 as nc
    # eikfile_nc = stem + ".eiknc.nc"
    eikfile_nc = eikfile + ".nc"

    print("Writing eikfile in netCDF format\n")

    ds0 = ds(eikfile_nc, "w")

    # The netCDF input file to GX doesn't take the last(repeated) element
    ntheta2 = ntheta - 1

    z_nc = ds0.createDimension("z", ntheta2)

    theta_nc = ds0.createVariable("theta", "f8", ("z",))
    theta_PEST_nc = ds0.createVariable("theta_PEST", "f8", ("z",))
    bmag_nc = ds0.createVariable("bmag", "f8", ("z",))
    gradpar_nc = ds0.createVariable("gradpar", "f8", ("z",))
    grho_nc = ds0.createVariable("grho", "f8", ("z",))
    gds2_nc = ds0.createVariable("gds2", "f8", ("z",))
    gds21_nc = ds0.createVariable("gds21", "f8", ("z",))
    gds22_nc = ds0.createVariable("gds22", "f8", ("z",))
    gbdrift_nc = ds0.createVariable("gbdrift", "f8", ("z",))
    gbdrift0_nc = ds0.createVariable("gbdrift0", "f8", ("z",))
    cvdrift_nc = ds0.createVariable("cvdrift", "f8", ("z",))
    cvdrift0_nc = ds0.createVariable("cvdrift0", "f8", ("z",))
    jacob_nc = ds0.createVariable("jacob", "f8", ("z",))

    Rplot_nc = ds0.createVariable("Rplot", "f8", ("z",))
    Zplot_nc = ds0.createVariable("Zplot", "f8", ("z",))
    aplot_nc = ds0.createVariable("aplot", "f8", ("z",))
    Rprime_nc = ds0.createVariable("Rprime", "f8", ("z",))
    Zprime_nc = ds0.createVariable("Zprime", "f8", ("z",))
    aprime_nc = ds0.createVariable("aprime", "f8", ("z",))

    drhodpsi_nc = ds0.createVariable(
        "drhodpsi",
        "f8",
    )
    kxfac_nc = ds0.createVariable(
        "kxfac",
        "f8",
    )
    Rmaj_nc = ds0.createVariable(
        "Rmaj",
        "f8",
    )
    q = ds0.createVariable(
        "q",
        "f8",
    )
    shat = ds0.createVariable(
        "shat",
        "f8",
    )

    theta_nc[:] = theta_eqarc[:-1]
    theta_PEST_nc[:] = theta_PEST[:-1]
    bmag_nc[:] = bmag_eqarc[:-1]
    gradpar_nc[:] = gradpar_eqarc[:-1]
    # add grho
    # grho_nc[:] = gds2_eqarc[:-1]
    gds2_nc[:] = gds2_eqarc[:-1]
    gds21_nc[:] = gds21_eqarc[:-1]
    gds22_nc[:] = gds22_eqarc[:-1]
    gbdrift_nc[:] = gbdrift_eqarc[:-1]
    gbdrift0_nc[:] = gbdrift0_eqarc[:-1]
    cvdrift_nc[:] = cvdrift_eqarc[:-1]
    cvdrift0_nc[:] = gbdrift0_eqarc[:-1]
    # jacob_nc[:]    = jacob_ball[:-1]

    Rplot_nc[:] = R_eqarc[:-1]
    Zplot_nc[:] = Z_eqarc[:-1]
    aplot_nc[:] = R_eqarc[:-1]

    Rprime_nc[:] = R_eqarc[:-1]
    Zprime_nc[:] = Z_eqarc[:-1]
    aprime_nc[:] = R_eqarc[:-1]

    drhodpsi_nc[0] = abs(1 / dpsidrho)
    kxfac_nc[0] = abs(qfac / rhoc * dpsidrho)
    Rmaj_nc[0] = (np.max(Rplot_nc) + np.min(Rplot_nc)) / 2
    q[0] = qfac
    # shat[0]        = shat

    ds0.close()
except ModuleNotFoundError:
    print(
        "No netCDF4 package in your Python environment...Not saving a netCDf input file"
    )
    pass
