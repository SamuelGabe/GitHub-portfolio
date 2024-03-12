#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 12:34:09 2021

@author: ppzjsb
"""

# Clear all for spyder
from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np
import matplotlib.pyplot as plt
#from sys import exit

# File directory and name
base = './planck1_20_1024_cold/'
file  = base+'los2048_n5000_z6.000.dat'

# Open the binary file
readdata = open(file,"rb")

# Header data
ztime  = np.fromfile(readdata,dtype=np.double,count=1) # redshift
omegaM = np.fromfile(readdata,dtype=np.double,count=1) # Omega_m (matter density)
omegaL = np.fromfile(readdata,dtype=np.double,count=1) # Omega_L (Lambda density)
omegab = np.fromfile(readdata,dtype=np.double,count=1) # Omega_b (baryon density)
h100   = np.fromfile(readdata,dtype=np.double,count=1) # Hubble constant, H0 / 100 km/s/Mpc
box100 = np.fromfile(readdata,dtype=np.double,count=1) # Box size in comoving kpc/h
Xh     = np.fromfile(readdata,dtype=np.double,count=1) # Hydrogen fraction by mass
nbins  = np.fromfile(readdata,dtype=np.int32,count=1)  # Number of pixels in each line of sight
numlos = np.fromfile(readdata,dtype=np.int32,count=1)  # Number of lines of sight

# Line of sight locations in box 
iaxis  = np.fromfile(readdata,dtype=np.int32,count=numlos[0])  # projection axis, x=1, y=2, z=3
xaxis  = np.fromfile(readdata,dtype=np.double,count=numlos[0]) # x-coordinate in comoving kpc/h
yaxis  = np.fromfile(readdata,dtype=np.double,count=numlos[0]) # y-coordinate in comoving kpc/h
zaxis  = np.fromfile(readdata,dtype=np.double,count=numlos[0]) # z-coordinate in comoving kpc/h

# Line of sight scale
posaxis = np.fromfile(readdata,dtype=np.double,count=nbins[0]) # comoving kpc/h
velaxis = np.fromfile(readdata,dtype=np.double,count=nbins[0]) # km/s

# Gas density, rho/<rho>
density = np.fromfile(readdata,dtype=np.double,count=nbins[0]*numlos[0])

# H1 fraction, fH1 = nH1/nH
H1frac  = np.fromfile(readdata,dtype=np.double,count=nbins[0]*numlos[0])

# Temperature, K
temp    = np.fromfile(readdata,dtype=np.double,count=nbins[0]*numlos[0])

# Peculiar velocity, km/s
vpec    = np.fromfile(readdata,dtype=np.double,count=nbins[0]*numlos[0])

# Close the binary file
readdata.close()


# Calculate clumping factor for gas with Delta<Dlim
Dlim = 100.0
ind = np.where(density <= Dlim)
clump = np.mean(density[ind]**2.0)/np.mean(density[ind])**2.0
print('Clumping factor',clump)


# Plot the gas density along a selected line of sight
PLOTLOS = 100 # select line of sight for plotting (0->numlos-1)
plt.figure(1, figsize=(12,3),constrained_layout=True)
ax = plt.gca()
plt.plot(velaxis,np.log10(density[PLOTLOS*nbins[0] : (PLOTLOS+1)*nbins[0]]),color='black')
plt.xlabel(r'$\rm v_{H}\,[km\, s^{-1}]$',fontsize=12)
plt.ylabel(r'log$_{10}(\rho/\langle \rho \rangle)$',fontsize=12)
plt.xlim([0,np.max(velaxis)])


# Plot the temperature-density plane for a random selection of npix pixels
npix = 100000
rnum = np.random.randint(0,nbins[0]*numlos[0],npix)

plt.figure(2, figsize=(5,4),constrained_layout=True)
ax = plt.gca()
plt.plot(np.log10(density[rnum]),np.log10(temp[rnum]),'.',color='red')
plt.xlim(-2,3)
plt.ylim(3,7)
plt.ylabel(r'log$_{10}(T/\rm K)$',fontsize=12)
plt.xlabel(r'log$_{10}(\rho/\langle \rho \rangle)$',fontsize=12)
plt.xlim([-1.5,2])
plt.ylim([3.5,5.5])





