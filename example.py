#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ARA AMB MATRIU DE COHERENCIA
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from COCOA import rw_matrix
import os
from scipy.special import laguerre

# Globals, em fa mandra fer-ho d'altra forma...
ny, nx = 512, 512
lamb = 500e-6
f = 5/lamb
Lf = 4
Lx = f*nx/4/Lf
Ly = Lx/nx*ny
NA = 0.95

def compute_save(tau, WS, dtype, ax=None, ylim=None, xlim=None):
    W = rw_matrix(tau, WS, Lx, Ly, f, NA, z=0, lamb=1, dtype=dtype, full_matrix=True)
    if ylim and xlim:
        W = W[ylim[0]:ylim[1], xlim[0]:xlim[1], :, :]
    W /= abs(W).max()
    Ix = np.real(W[:, :, 0, 0])
    Iy = np.real(W[:, :, 1, 1])
    Iz = np.real(W[:, :, 2, 2])
    It = Ix+Iy+Iz
    
    maxval = It.max()
    minval = 0

    # 3D Degree of Polarization according to Setala
    tr2 = np.trace(W@W, axis1=-2, axis2=-1)
    P = np.real(np.sqrt(3/2*(tr2/It**2-1/3)))
    
    cmap = "gist_heat"
    sz = 3.5
    if type(ax)==None:
        fig, ax = plt.subplots(1, 4, figsize=(4*sz, sz), constrained_layout=True)
    ax[0].imshow(Ix+Iy, cmap=cmap, extent=[-Lf, Lf, -Lf, Lf], vmin=minval, vmax=It.max())

    ax[1].imshow(Iz, cmap=cmap, extent=[-Lf, Lf, -Lf, Lf], vmin=minval, vmax=It.max())

    imsh = ax[2].imshow(It, cmap=cmap, extent=[-Lf, Lf, -Lf, Lf], vmin=minval, 
            vmax=It.max())

    pol = ax[3].imshow(P, cmap="bwr", extent=[-Lf, Lf, -Lf, Lf], vmin=minval, vmax=1)

    print("\t", P.min(), P.max())
    return imsh, pol

def plot_coherence(L_EP, n, *sigmes, labels=["Coherent", "Incoherent", 
    "Partially coherent"], fontsize=16):
    x = np.linspace(-L_EP, L_EP, 511)
    x2 = x*x
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for i, sig in enumerate(sigmes):
        a = np.exp(-x2*.5/sig**2)
        coh = a*laguerre(n)(x2/sig**2*.5)
        ax[1].plot(x/x.max(), coh, label=labels[i])
    ax[1].legend(loc=(0.6, 0.6), fontsize=fontsize)
    ax[1].set_xlabel(r"$\Delta/w_0$", fontsize=fontsize)
    ax[1].set_ylabel(r"Degree of coherence", fontsize=fontsize)
    ax[1].tick_params(axis="both", labelsize=fontsize)
    ax[1].set_title("(b)", fontsize=fontsize)

    # Amplitude
    ax[0].plot(x/x.max(), (np.exp(-x2**2/L_EP**2)*abs(x/L_EP))**2)
    ax[0].set_xlabel(r"$\rho/w_0$", fontsize=fontsize)
    ax[0].set_ylabel(r"Irradiance (a. u.)", fontsize=fontsize)
    ax[0].tick_params(axis="both", labelsize=fontsize)
    ax[0].set_title("(a)", fontsize=fontsize)

    fig.savefig("coherences.pdf", bbox_inches="tight", dpi=200)

if __name__ == "__main__":
    dtype = np.complex_
    # Padding properties
    npy = 3*ny
    npx = 3*nx
    ny_min = npy//2-ny//2
    ny_max = npy//2+ny//2
    nx_min = npx//2-nx//2
    nx_max = npx//2+nx//2

    # Matrius de coherencia
    y, x = np.mgrid[-ny//2:ny//2, -nx//2:nx//2]
    phi = np.arctan2(y, x)
    r2 = x*x+y*y
    r = np.sqrt(r2)
    # Calculate entrance pupil in pixels!
    pixel_per_dist = nx/(2*Lx)
    L_EP = f*NA/Lx*nx*2
    # Tau matrix contains the amplitudes of the field
    tau = np.zeros((npy, npy, 2, 2), dtype=dtype)
    tau[ny_min:ny_max, nx_min:nx_max, 0, 0] = np.cos(phi)*r*np.exp(-r2/2/L_EP**2)
    tau[ny_min:ny_max, nx_min:nx_max, 1, 1] = np.sin(phi)*r*np.exp(-r2/2/L_EP**2)
    polaritzacions = {
            "radial": (np.asarray([[1, 1], 
                                   [1, 1]], dtype=dtype), tau),
            }
    sig_coh = L_EP*1e3
    sig_incoh = L_EP/50
    sig_parcial = L_EP/4
    n = 5
    # Plot coherences and amplitudes
    labels = [r"$\mu\rightarrow \infty$",
              r"$\mu = w_0/50$",
              r"$\mu = w_0/4$"]
    plot_coherence(L_EP, n, sig_coh, sig_incoh, sig_parcial,
            labels=labels)
    coherencies = {
            "coherent": laguerre(n)(r2/sig_coh**2*.5)*np.exp(-.5*r2/sig_coh**2),
            "parcial": laguerre(n)(r2/sig_parcial**2*.5)*np.exp(-.5*r2/sig_parcial**2),
            "incoherent": laguerre(n)(r2/sig_incoh**2*.5)*np.exp(-.5*r2/sig_incoh**2),
            }

    # Create an auxiliary H matrix, which contains the degree of coherence of the beam
    H = np.zeros((npy, npx, 2, 2), dtype=dtype)
    for pol_kind in polaritzacions:
        P, t = polaritzacions[pol_kind]
        print(pol_kind)
        fig, ax = plt.subplots(3, 4, figsize=(10,7), constrained_layout=True)
        pcm_irr = []
        pcm_pol = []
        for i, coherency in enumerate(coherencies):
            h = coherencies[coherency]
            # Insert the degree of coherence on the H matrix
            H[ny_min:ny_max, nx_min:nx_max, 0, 0] = h
            H[ny_min:ny_max, nx_min:nx_max, 1, 1] = h
            # Compute the WS matrix as the matrix product of P and H
            WS = P@H
            # Finally, compute the focal field
            p1, p2 = compute_save(t, WS, dtype, 
                    ax=ax[i,:], ylim=(ny_min, ny_max), xlim=(ny_min, ny_max))
            pcm_irr.append(p1)
            pcm_pol.append(p2)
        
        # Plot the values of the irradiances and the DoP
        for j in range(3):
            for i in range(4):
                ax[j, i].set_xlabel(r"$x(\lambda)$")
                ax[j, i].set_ylabel(r"$y(\lambda)$")
        ax[0, 0].set_title("$\hat{W}_{11}+\hat{W}_{22}$")
        ax[0, 1].set_title("$\hat{W}_{33}$")
        ax[0, 2].set_title("Tr$\hat{W}$")
        ax[0, 3].set_title("DoP")
        fs = 16
        ax[0, 0].text(3.5, -3, r"$\mu \rightarrow \infty$", color="white", 
                horizontalalignment="right", size=fs)
        ax[1, 0].text(3.5, -3, r"$\mu = \frac{w_0}{4}$", color="white", 
                horizontalalignment="right", size=fs)
        ax[2, 0].text(3.5, -3, r"$\mu = \frac{w_0}{50}$", color="white", 
                horizontalalignment="right", size=fs)
        # Colorbars
        fig.colorbar(pcm_irr[0], ax=ax[:, 2], shrink=.919, aspect=80)
        fig.colorbar(pcm_pol[0], ax=ax[:, 3], shrink=.919, aspect=80)
        fig.savefig("pol_matrix_ctant.pdf", bbox_inches="tight", dpi=200)
        fig.savefig("pol_matrix_ctant.png", bbox_inches="tight", dpi=200)
        plt.show()
        plt.close("all")
