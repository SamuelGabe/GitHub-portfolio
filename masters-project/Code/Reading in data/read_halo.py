#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 16:56:09 2022

@author: ppzjsb
"""

import numpy as np
import matplotlib.pyplot as plt
#from sys import exit

h        = 0.678 # H0/100 km/s/Mpc
Lbox     = 20000.0 # box size in ckpc/h
mdm_part = 5.37e5 # dm particle mass in Msol/h

def calc_hmf(dmm, z):
    file = f'Data.nosync/planck1_20_1024_{dmm}/halolist_z{z:.3f}.dat'

    # Open the binary file
    readdata = open(file,"rb")

    # Header data
    ngroups = np.fromfile(readdata,dtype=np.int32,count=1)  # number of FoF groups (haloes)

    CMx    = np.fromfile(readdata, dtype=np.float32, count=ngroups[0]) # x-coordinate halo centre of mass, comoving kpc/h
    CMy    = np.fromfile(readdata, dtype=np.float32, count=ngroups[0]) # y-coordinate halo centre of mass, comoving kpc/h
    CMz    = np.fromfile(readdata, dtype=np.float32, count=ngroups[0]) # z-coordinate halo centre of mass, comoving kpc/h
    Mgas   = np.fromfile(readdata, dtype=np.float32, count=ngroups[0]) # gas mass, Msol/h
    Mdm    = np.fromfile(readdata, dtype=np.float32, count=ngroups[0]) # dark matter mass, Msol/h
    Mstars = np.fromfile(readdata, dtype=np.float32, count=ngroups[0]) # stellar mass, Msol/h

    readdata.close()

    CMr = np.sqrt(CMx*CMx + CMy*CMy + CMz*CMz)

    print('Total groups:',ngroups[0])

    mass_limit = np.log10(32.0 * mdm_part)
    ind = np.where(np.log10(Mdm) >= mass_limit)

    print('Total resolved groups:', len(Mdm[ind]))

    # Compute and plot dark matter halo mass function
    nbins    = 25
    binmin   = np.log10(np.min(Mdm))
    binmax   = np.log10(np.max(Mdm))
    binsize  = (binmax - binmin) / nbins

    massfn,mbin_edge = np.histogram(np.log10(Mdm), bins=np.linspace(binmin,binmax,nbins+1))

    # Calculate bin centres and normalise the distribution
    mbin   = binmin + np.linspace(0,nbins-1,nbins)*binsize + 0.5*binsize
    massfn = massfn / (binsize*(Lbox/1.0e3)**3.0)
    
    return mbin, massfn, mass_limit

# Z array is defined by files. dmm is in each filename
dmmlist = ['lambda', 'cold', 'hot', 'wdm1', 'wdm2', 'wdm3', 'wdm4']
zarr    = np.array([4.200, 4.800, 5.400, 6.000, 7.000, 8.000, 10.000])
stem    = 'Data.nosync/Halo mass data/'
z = 6.0

mass_limit = 0

# Save different file for each dark matter model (dmm)
for dmm in dmmlist:
    # TODO: Figure out what to plot the halo mass function against.

    mbin, massfn, mass_limit = calc_hmf(dmm, z)                            # Create halo mass array
    # data = np.array([zarr, MHarr])                                                  # Package up to be saved in a txt file
    # np.save(stem + dmm, data)
    plt.plot(np.log10(mbin), np.log10(massfn), label=dmm, linewidth=1.0)

plt.plot(np.log10([mass_limit,mass_limit]), [-4,4], '--')
plt.legend()
# plt.savefig(f'Halo mass graph, z={z}')

# ndot gamma
nion_filename = 'Data.nosync/Ndotion data/nion_kulkarni.txt'
nion = np.loadtxt(nion_filename)
nion6_idx = np.where(nion[:,0] == 6.0)
nion6 = nion[nion6_idx, 0]

# volume of box = length cubed
vbox = Lbox ** 3
SUM_MASSES = 0          # REPLACE
ngamma = nion6 / vbox / SUM_MASSES

plt.show()
