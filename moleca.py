#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from scipy.fft import fft2, ifft2
fftshift = np.fft.fftshift
ifftshift = np.fft.ifftshift

def convolve(f, g):
    """Convolve two 2D functions f, g."""
    fk = fft2(f, workers=-1)
    gk = fft2(g, workers=-1)
    return ifft2(gk*np.conj(fk), workers=-1)

def rw_matrix(tau, WS, Lx, Ly, f, NA, z=0, lamb=500e-6, dtype=np.complex64,
        full_matrix=False):
    """
    Computes the CSDM of a highly focused Electromagnetic Schell
    model beam based on the Moleca algorithm [reference pending].

    Input:
        - tau: (Ny, Nx, 2, 2) matrix, containing the amplitudes of the input beam.
        - WS: (Ny, Nx, 2, 2) matrix with both the polarization and degrees of coherence
        of the input beam.
        - Lx: Half size of the sampling window in the X direction.
        - Ly: Half size of the sampling window in the Y direction.
        - f: Focal length of the optical system.
        - NA: Numerical aperture of the optical system.
        - z (optional): distance from the focal plane.
        - lamb (optional): Wavelength of the light
        - dtype (optional): Precision type of the operations. Must be either complex128 or
        complex64.
        - full_matrix (optional): By default, only the upper triangle components are
        computed. If this is True, the lower triangle of CSDM is populated with the
        complex conjugates of the upper triangle.

    Output:
        - CSDM: (Ny, Nx, 3, 3) matrix containing the CSDM of the highly focused
        electromagnetic beam.

    """
    # Select precision
    if dtype != np.complex64 and dtype != np.complex_:
        raise ValueError("Precision type not understood.")

    ny, nx, _, _ = tau.shape
    y, x = np.mgrid[-ny//2:nx//2, -nx//2:nx//2]
    phi = np.arctan2(y, x)
    Lxf = lamb*f*nx/4/Lx
    Lyf = lamb*f*ny/4/Ly
    thmax = np.arcsin(NA)
    #d = f*np.tan(thmax)#/2   # Entre dos?????
    d = f*NA/np.sqrt(1-NA*NA)

    # Coordinates at the GRS
    x_inf = x/nx*Lx*2
    y_inf = y/ny*Ly*2
    rho2 = x_inf*x_inf+y_inf*y_inf

    # Cosine and sine factors and E.P. mask
    cosphi = np.cos(phi)
    sinphi = np.sin(phi)
    sinth2 = rho2/f/f
    sinth = np.sqrt(sinth2)
    mask = sinth < NA
    costh = np.sqrt(1-sinth2*mask)
    sqcos = np.sqrt(costh)

    # Build the N0 matrix
    N = np.zeros((ny, nx, 2, 3), dtype=dtype)
    N[:, :, 0, 0] = (sinphi*sinphi + cosphi*cosphi*costh)
    N[:, :, 0, 1] = sinphi*cosphi*(costh-1)
    N[:, :, 0, 2] = sinth*cosphi

    N[:, :, 1, 0] = sinphi*cosphi*(costh-1)
    N[:, :, 1, 1] = (sinphi*sinphi*costh + cosphi*cosphi)
    N[:, :, 1, 2] = sinth*sinphi

    N0 = np.matmul(tau, N)
    for j in range(2):
        for i in range(3):
            N0[:, :, j, i] = N0[:, :, j, i]/sqcos*np.exp(-2j*np.pi/lamb*z*costh)*mask

    # Create the coherency matrix and calculate its components
    W = np.zeros((ny, nx, 3, 3), dtype=dtype)
    # ONLY compute upper diagonal terms
    Wshift = fftshift(WS, axes=(0, 1))
    for j in range(3):
        for i in range(j, 3):
            W[:, :, j, i] = fft2(
                    convolve(N0[:, :, 0, j], N0[:, :, 0, i])*Wshift[:, :, 0, 0]+\
                    convolve(N0[:, :, 1, j], N0[:, :, 1, i])*Wshift[:, :, 1, 1]+\
                    convolve(N0[:, :, 0, j], N0[:, :, 1, i])*Wshift[:, :, 0, 1]+\
                    convolve(N0[:, :, 1, j], N0[:, :, 0, i])*Wshift[:, :, 1, 0],
                    workers=-1)

    if full_matrix:
        W[:, :, 1, 0] = np.conj(W[:, :, 0, 1])
        W[:, :, 2, 0] = np.conj(W[:, :, 0, 2])
        W[:, :, 2, 1] = np.conj(W[:, :, 1, 2])
    return fftshift(W, axes=(0, 1))
