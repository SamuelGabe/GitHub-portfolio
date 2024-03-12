import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.constants as con

sqrt = np.sqrt
G = con.G

zarr = np.linspace(20, 0, 1000)

# 1 MPc in metres
MPc = 3.0857E+22

# Define constants. Some are taken from the simulations
ndot_ion = 3E+50 / MPc**3
mass_H   = 1.67262192E-27
clump_H = 3
Xh = 0.76
Yhe = 1 - Xh
temp = 1E4
H0 = 67.5 * 1000 / MPc

omegab = 0.0482
omegaL = 0.692
omegaM = 0.308

# Calculate alpha
a = 7.982e-11; b = 0.7480; T0 = 3.148e+00; T1 = 7.036e+05

# Divide symbol at the start instead of putting **-1 at the end
alpha = a / (sqrt(temp/T0) * (1 + sqrt(temp/T0))**(1-b) * (1+sqrt(temp/T1))**(1+b))
alpha /= 1e6

# Average number density of hydrogen. (1+z)**3 removed because nion factor has (1+z)**3 factor as well
n_avg = (3 * (H0*H0) / (8 * np.pi * G)) * omegab * Xh / mass_H

# Functions to work out non-constant values
def hubble(z): return H0 * sqrt(omegaM * (1+z)**3 + omegaL)
def trec(z): return 1 / (clump_H * alpha * n_avg * (1 + Yhe / (4 * Xh)) * (1+z)**3)

# Returns dQ/dz given Q and z
def dQdz(Q, z):
    return 1 / (hubble(z) * (1 + z)) * (Q / trec(z) - ndot_ion / n_avg)

# Initialise Q array
Qarr = np.zeros_like(zarr)

for count, z in enumerate(zarr):
    if count < len(zarr) - 1:
        Q0 = Qarr[count]
        dz = zarr[count+1] - zarr[count]
        #Qarr[count+1] = (Q0 - ndot_ion * dz / (n_avg * hubble(z) * (1 + z))) / (1 - dz / (hubble(z) * (1+z) * trec(z)))
        #Qarr[count+1] = Qarr[count] + ((dz / (hubble(z) * (1 + z))) * (Qarr[count] / trec(z) - ndot_ion / n_avg))
        Qarr[count+1] = Qarr[count] + dz * dQdz(Qarr[count], z)


plt.figure(figsize=(7, 5))
plt.plot(zarr, Qarr, 'k')
plt.xlim(0, np.max(zarr))
plt.ylim(0., 1.)
plt.xlabel('$z$')
plt.ylabel('$Q$')
# plt.savefig('Graphs/Important Graphs/Forward Euler C=3 14-03.png', dpi=300)
plt.show()