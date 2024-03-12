import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import scipy.constants as con

sqrt = np.sqrt
G = con.G

zarr = np.linspace(20, 0, 100)

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

# First divide is instead of putting **-1 at the end
alpha = a / (sqrt(temp/T0) * (1 + sqrt(temp/T0))**(1-b) * (1+sqrt(temp/T1))**(1+b))

# Functions to work out non-constant values
def n_avg(z): return ((3 * (H0*H0) / (8 * np.pi * G)) * omegab * Xh * (1+z)**3) / mass_H
def hubble(z): return H0 * sqrt(omegaM * (1+z)**3 + omegaL)
def trec(z): return 1 / (clump_H * alpha * n_avg(z) * (1 + Yhe / (4 * Xh)) * (1+z)**3)

def dQdz(z, Q):
    return 1 / (hubble(z) * (1 + z)) * (Q / trec(z) - ndot_ion / n_avg(z))

Qarr = np.zeros_like(zarr)

solution = solve_ivp(dQdz, [20.0, 0.0], [0.0])

plt.plot(solution.t, solution.y)
plt.show()